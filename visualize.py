"""
visualize.py — Four-panel pipeline visualizer.

Each card shows a 2×2 grid:
  ┌──────────────────┬──────────────────┐
  │   Pyomo code     │   Pyomo math     │
  ├──────────────────┼──────────────────┤
  │   Gurobi math    │   Gurobi code    │
  └──────────────────┴──────────────────┘

Cross-panel hover/click highlighting by semantic class (set, var, param, constraint).
Static: no code execution, no fake-gurobipy. Everything derived from ASTs.

Usage:
    from visualize import visualize
    import examples.example_4_bom as ex
    visualize(ex.build_pyomo_model, needs_approval=True)
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
import textwrap
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional


# ===================================================================
# 1. Spec extraction (AST walk of build_pyomo_model)
# ===================================================================

@dataclass
class SetSpec:
    name: str
    data_key: str
    dimen: int = 1
    is_indexed: bool = False
    index_set: Optional[str] = None
    is_subset: bool = False
    within_set: Optional[str] = None


@dataclass
class ParamSpec:
    name: str
    data_key: str
    index_sets: list[str] = field(default_factory=list)


@dataclass
class VarSpec:
    name: str
    index_sets: list[str] = field(default_factory=list)
    vtype: str = 'CONTINUOUS'


@dataclass
class ConstrSpec:
    name: str
    index_sets: list[str]
    rule_args: list[str]
    rule_body_node: Optional[ast.AST]
    rule_assigns: dict[str, ast.expr] = field(default_factory=dict)
    op: Optional[str] = None
    is_inline: bool = False
    raw_source: str = ''


@dataclass
class ObjSpec:
    name: str
    sense: str
    rule_args: list[str]
    rule_body_node: Optional[ast.AST]
    is_inline: bool = False
    raw_source: str = ''


@dataclass
class ModelSpec:
    sets: dict[str, SetSpec] = field(default_factory=dict)
    params: dict[str, ParamSpec] = field(default_factory=dict)
    vars: dict[str, VarSpec] = field(default_factory=dict)
    constrs: list[ConstrSpec] = field(default_factory=list)
    obj: Optional[ObjSpec] = None
    func_source: str = ''
    _all_data_keys: list = field(default_factory=list)


def _is_m_attr(node) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'm':
        return node.attr
    return None


def _is_data_subscript(node) -> Optional[str]:
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == 'data'):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return None


def _is_data_method_call(node) -> Optional[str]:
    while isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {'list', 'tuple', 'set'}:
            if not node.args:
                return None
            node = node.args[0]
        elif (isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Subscript)
              and isinstance(node.func.value.value, ast.Name)
              and node.func.value.value.id == 'data'):
            sl = node.func.value.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return sl.value
            return None
        else:
            return None
    return _is_data_subscript(node)


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _all_data_keys(tree: ast.AST) -> list[str]:
    keys = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'data'
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.append(node.slice.value)
    return keys


def _ast_op_to_str(op) -> str:
    return {ast.LtE: '<=', ast.GtE: '>=', ast.Eq: '==',
            ast.Lt: '<', ast.Gt: '>'}.get(type(op), '?')


def extract_spec(build_pyomo_fn) -> ModelSpec:
    raw = inspect.getsource(build_pyomo_fn)
    src = textwrap.dedent(raw)
    tree = ast.parse(src)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a top-level function definition.")

    spec = ModelSpec(func_source=src)
    spec._all_data_keys = _all_data_keys(func_def)
    rules: dict[str, ast.FunctionDef] = {}

    for stmt in func_def.body:
        if isinstance(stmt, ast.FunctionDef):
            rules[stmt.name] = stmt

    for stmt in func_def.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == 'm'):
            continue
        attr = tgt.attr
        val = stmt.value
        if not isinstance(val, ast.Call):
            continue
        f = val.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == 'pyo'):
            continue
        kind = f.attr

        if kind == 'Set':
            spec.sets[attr] = _parse_set(attr, val)
        elif kind == 'Param':
            spec.params[attr] = _parse_param(attr, val)
        elif kind == 'Var':
            spec.vars[attr] = _parse_var(attr, val)
        elif kind == 'Constraint':
            spec.constrs.append(_parse_constraint(attr, val, rules, src))
        elif kind == 'Objective':
            spec.obj = _parse_objective(attr, val, rules, src)

    return spec


def _parse_set(name: str, call: ast.Call) -> SetSpec:
    init = _kw(call, 'initialize')
    data_key = None
    if init is not None:
        data_key = _is_data_subscript(init) or _is_data_method_call(init)
    dimen_kw = _kw(call, 'dimen')
    dimen = dimen_kw.value if isinstance(dimen_kw, ast.Constant) else 1
    is_indexed = False
    index_set = None
    if call.args:
        a0 = call.args[0]
        if (m_attr := _is_m_attr(a0)):
            is_indexed = True
            index_set = m_attr
    is_subset = False
    within_set = None
    within = _kw(call, 'within')
    if within is not None:
        if (m_attr := _is_m_attr(within)):
            is_subset = True
            within_set = m_attr
    return SetSpec(name=name, data_key=data_key or name, dimen=dimen,
                   is_indexed=is_indexed, index_set=index_set,
                   is_subset=is_subset, within_set=within_set)


def _parse_param(name: str, call: ast.Call) -> ParamSpec:
    idx_sets = []
    for a in call.args:
        if (m_attr := _is_m_attr(a)):
            idx_sets.append(m_attr)
    init = _kw(call, 'initialize')
    data_key = _is_data_subscript(init) if init is not None else None
    return ParamSpec(name=name, data_key=data_key or name, index_sets=idx_sets)


def _parse_var(name: str, call: ast.Call) -> VarSpec:
    idx_sets = []
    for a in call.args:
        if (m_attr := _is_m_attr(a)):
            idx_sets.append(m_attr)
    vtype = 'CONTINUOUS'
    domain = _kw(call, 'domain')
    if domain is not None:
        attr = None
        if isinstance(domain, ast.Attribute):
            attr = domain.attr
        if attr:
            if 'Integer' in attr:
                vtype = 'INTEGER'
            elif 'Binary' in attr:
                vtype = 'BINARY'
    return VarSpec(name=name, index_sets=idx_sets, vtype=vtype)


def _parse_constraint(name: str, call: ast.Call, rules: dict, src: str) -> ConstrSpec:
    idx_sets = []
    for a in call.args:
        if (m_attr := _is_m_attr(a)):
            idx_sets.append(m_attr)

    rule_kw = _kw(call, 'rule')
    expr_kw = _kw(call, 'expr')
    rule_args, rule_body_node, rule_assigns, raw_source = [], None, {}, ''
    is_inline = False
    op = None

    if rule_kw is not None and isinstance(rule_kw, ast.Name) and rule_kw.id in rules:
        rule_def = rules[rule_kw.id]
        args = [a.arg for a in rule_def.args.args]
        rule_args = [a for a in args if a != 'm']
        for stmt in rule_def.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                rule_assigns[stmt.targets[0].id] = stmt.value
            elif isinstance(stmt, ast.Return):
                rule_body_node = stmt.value
        try:
            raw_source = textwrap.dedent(ast.get_source_segment(src, rule_def) or '')
        except Exception:
            raw_source = ''
        if isinstance(rule_body_node, ast.Compare) and rule_body_node.ops:
            op = _ast_op_to_str(rule_body_node.ops[0])
    elif expr_kw is not None:
        is_inline = True
        rule_body_node = expr_kw
        try:
            raw_source = ast.get_source_segment(src, call) or ''
        except Exception:
            raw_source = ''
        if isinstance(expr_kw, ast.Compare) and expr_kw.ops:
            op = _ast_op_to_str(expr_kw.ops[0])

    return ConstrSpec(name=name, index_sets=idx_sets, rule_args=rule_args,
                      rule_body_node=rule_body_node, rule_assigns=rule_assigns,
                      op=op, is_inline=is_inline, raw_source=raw_source)


def _parse_objective(name: str, call: ast.Call, rules: dict, src: str) -> ObjSpec:
    sense_kw = _kw(call, 'sense')
    sense = 'MINIMIZE'
    if sense_kw is not None:
        attr = None
        if isinstance(sense_kw, ast.Attribute):
            attr = sense_kw.attr
        if attr and 'maximize' in attr.lower():
            sense = 'MAXIMIZE'

    rule_kw = _kw(call, 'rule')
    expr_kw = _kw(call, 'expr')
    rule_args, rule_body_node, raw_source = [], None, ''
    is_inline = False

    if rule_kw is not None and isinstance(rule_kw, ast.Name) and rule_kw.id in rules:
        rdef = rules[rule_kw.id]
        rule_args = [a.arg for a in rdef.args.args if a.arg != 'm']
        for stmt in rdef.body:
            if isinstance(stmt, ast.Return):
                rule_body_node = stmt.value
        try:
            raw_source = textwrap.dedent(ast.get_source_segment(src, rdef) or '')
        except Exception:
            raw_source = ''
    elif expr_kw is not None:
        is_inline = True
        rule_body_node = expr_kw
        try:
            raw_source = ast.get_source_segment(src, call) or ''
        except Exception:
            raw_source = ''

    return ObjSpec(name=name, sense=sense, rule_args=rule_args,
                   rule_body_node=rule_body_node, is_inline=is_inline,
                   raw_source=raw_source)


# ===================================================================
# 2. Gurobi code slice extraction
# ===================================================================

def extract_gurobi_slices(generated_code: str) -> dict[str, str]:
    """Split generated code into per-constraint slices using comment markers."""
    lines = generated_code.splitlines()
    slices: dict[str, list] = {}
    current: Optional[str] = None

    for line in lines:
        s = line.strip()
        if s.startswith('# Constraint:'):
            current = s[len('# Constraint:'):].strip()
            slices.setdefault(current, [])
        elif s.startswith('# Objective'):
            current = '__objective__'
            slices.setdefault(current, [])
        elif current is not None:
            # Stop slice at end of function (m._get_values or return)
            if s.startswith('m._get_values') or s.startswith('return m'):
                current = None
            else:
                slices[current].append(line)

    return {k: '\n'.join(v).strip() for k, v in slices.items() if v}


# ===================================================================
# 2b. Gurobi micro-section classifier
# ===================================================================

_SECTION_LABELS = {
    'define_rows':   '▸ Define constraint rows',
    'map_coeffs':    '▸ Map coefficients',
    'build_matrix':  '▸ Build sparse matrix',
    'register':      '▸ Register with Gurobi',
    'save_order':    '▸ Save row order',
}


def _gurobi_section(line: str) -> str:
    """Classify a line of generated Gurobi code into a micro-section."""
    s = line.strip()
    if not s:
        return 'blank'
    if '_rhs_ord' in s and '=' in s:
        return 'save_order'
    if '.addMConstr(' in s or ('.addConstr(' in s and 'm._mconstr' in s):
        return 'register'
    if 'csr_matrix' in s:
        return 'build_matrix'
    # _constr_fwd_* and _constr_bwd_* are rename steps — part of coefficient mapping
    if re.match(r'\s*_constr_(fwd|bwd)_', line):
        return 'map_coeffs'
    # Primary row-definition patterns
    if (re.match(r'\s*_constr_\w+\s*=', line) or
            re.match(r'\s*_idx_\w+\s*=', line) or
            'm._constr_idx' in s):
        return 'define_rows'
    # Coefficient mapping patterns
    if any(p in s for p in ('pd.merge', 'pd.concat', '_fp_', '_m1_', '_coo_',
                             '_fwd_', '_bwd_', '_dc_')):
        return 'map_coeffs'
    return 'define_rows'


def _find_coo_value_col(generated_code: str, constr_name: str) -> Optional[str]:
    """Find the non-index value column used in csr_matrix for this constraint."""
    # Match only the data (non-row/col) column: csr_matrix((frame['col'].values, ...)
    # Negative lookahead skips _row and _col which are index coordinates, not values
    pat = rf"_coo_{re.escape(constr_name)}\['(?!_row|_col)(\w+)'\]\.values"
    m = re.search(pat, generated_code)
    return m.group(1) if m else None


# ===================================================================
# 2c. Synthetic data generation
# ===================================================================

_SYN_LABELS = ['A', 'B']


def _build_synthetic_data(spec: ModelSpec, build_pyomo_fn) -> Optional[dict]:
    """Generate canonical synthetic data ({A, B} per 1-D set) for the model.

    Uses real data (if available on the module) to peek at structural shapes
    (e.g. 'is the inner of this indexed family a list of 3-tuples?') and
    generates canonical {A, B} labels in those shapes.

    Returns a `data` dict, or None if synthesis fails. If the module exposes
    `preprocess_data`, it is invoked on the synthesized raw to fill derived keys.
    """
    raw: dict = {}

    # Peek at module.data for shape inference
    import inspect
    module = inspect.getmodule(build_pyomo_fn)
    real_data = getattr(module, 'data', None) if module else None

    def _inner_arity_of(data_key: str) -> int:
        """For an indexed family set, peek at real data to find inner-tuple arity.

        Scans all values for the first non-empty list to determine arity, so
        sparse keys whose first value is `[]` don't confuse inference.
        """
        if not real_data or data_key not in real_data:
            return 1
        v = real_data[data_key]
        if not isinstance(v, dict):
            return 1
        for sample in v.values():
            if isinstance(sample, list) and sample:
                first = sample[0]
                if isinstance(first, tuple):
                    return len(first)
                return 1
        return 1

    def _make_inner(arity: int) -> list:
        """Two canonical inner values for an indexed family member."""
        if arity == 1:
            return list(_SYN_LABELS)
        # arity-N: produce 2 distinct N-tuples from canonical labels
        a, b = _SYN_LABELS[0], _SYN_LABELS[1]
        t1 = tuple([a if i % 2 == 0 else b for i in range(arity)])
        t2 = tuple([b if i % 2 == 0 else a for i in range(arity)])
        return [t1, t2]

    # 1-D non-subset sets
    for sname, sspec in spec.sets.items():
        if sspec.is_indexed or sspec.dimen != 1 or sspec.is_subset:
            continue
        raw[sspec.data_key] = list(_SYN_LABELS)

    # subset sets: pick the first synthetic label of the parent set
    for sname, sspec in spec.sets.items():
        if not sspec.is_subset:
            continue
        parent = spec.sets.get(sspec.within_set)
        parent_vals = (raw.get(parent.data_key, _SYN_LABELS)
                       if parent else _SYN_LABELS)
        raw[sspec.data_key] = [parent_vals[0]]

    # 2-tuple sparse sets (e.g., Edges, dimen=2)
    for sname, sspec in spec.sets.items():
        if sspec.dimen != 2 or sspec.is_indexed:
            continue
        raw[sspec.data_key] = [(_SYN_LABELS[0], _SYN_LABELS[1]),
                               (_SYN_LABELS[1], _SYN_LABELS[0])]

    # indexed family sets (Set[Z])
    for sname, sspec in spec.sets.items():
        if not sspec.is_indexed:
            continue
        idx_set_spec = spec.sets.get(sspec.index_set)
        idx_vals = (raw.get(idx_set_spec.data_key, _SYN_LABELS)
                    if idx_set_spec else _SYN_LABELS)
        arity = _inner_arity_of(sspec.data_key)
        raw[sspec.data_key] = {z: _make_inner(arity) for z in idx_vals}

    # parameters: enumerate keys from index sets, fill distinct integers
    val_counter = [1]

    def _next_val():
        v = val_counter[0]
        val_counter[0] += 1
        return v

    def _enum_indices(idx_sets):
        """Yield index tuples for the Cartesian product of given set names."""
        if not idx_sets:
            yield ()
            return
        first, rest = idx_sets[0], idx_sets[1:]
        sspec = spec.sets.get(first)
        if sspec is None:
            for v in _SYN_LABELS:
                for tail in _enum_indices(rest):
                    yield (v,) + tail
            return
        if sspec.dimen == 2 and not sspec.is_indexed:
            outer = raw.get(sspec.data_key, [])
            for tup in outer:
                for tail in _enum_indices(rest):
                    yield tup + tail
            return
        outer = raw.get(sspec.data_key, _SYN_LABELS)
        for v in outer:
            for tail in _enum_indices(rest):
                yield (v,) + tail

    for pname, pspec in spec.params.items():
        if not pspec.index_sets:
            raw[pspec.data_key] = 10
            continue
        keys = list(_enum_indices(pspec.index_sets))
        if not keys:
            raw[pspec.data_key] = {}
        elif all(len(k) == 1 for k in keys):
            raw[pspec.data_key] = {k[0]: _next_val() for k in keys}
        else:
            raw[pspec.data_key] = {k: _next_val() for k in keys}

    # Scalars referenced inline (e.g., `data['Budget']` in expr=)
    bound = {sspec.data_key for sspec in spec.sets.values()}
    bound.update(p.data_key for p in spec.params.values())
    for k in spec._all_data_keys:
        if k not in raw and k not in bound:
            raw[k] = 10

    # If module provides preprocess_data, run it (fills derived keys)
    import inspect
    module = inspect.getmodule(build_pyomo_fn)
    preprocess = getattr(module, 'preprocess_data', None) if module else None
    data = raw
    if callable(preprocess):
        try:
            result = preprocess(dict(raw))
            if isinstance(result, dict):
                data = result
        except Exception:
            data = raw  # preprocess failed; raw may still be sufficient

    return data


# ===================================================================
# 2d. Mock execution — capture all DataFrames produced by generated code
# ===================================================================

def _make_mock_gp():
    import types

    class _MockMVar:
        X = None
        def __matmul__(self, other): return 0
        def __rmatmul__(self, other): return 0
        def __add__(self, other): return self
        def __radd__(self, other): return self

    class _MockSparse:
        def __matmul__(self, other): return 0
        def __add__(self, other): return self
        def __radd__(self, other): return self

    class _MockModel:
        def __init__(self):
            self._mconstr = {}
            self._rhs_ord = {}
            self._constr_idx = {}
            self._mvars = {}
            self._var_idx = {}
        def addMVar(self, n, **kw): return _MockMVar()
        def addMConstr(self, *a, **kw): return None
        def addConstr(self, *a, **kw): return None
        def setObjective(self, *a, **kw): pass
        def remove(self, *a): pass

    GRB = types.SimpleNamespace(
        MINIMIZE=-1, MAXIMIZE=1,
        LESS_EQUAL='<', GREATER_EQUAL='>', EQUAL='=',
        CONTINUOUS='C', INTEGER='I', BINARY='B',
    )
    mock_sparse = types.SimpleNamespace(
        csr_matrix=lambda *a, **kw: _MockSparse()
    )
    mock_scipy = types.SimpleNamespace(sparse=mock_sparse)
    mock_gp = types.SimpleNamespace(Model=_MockModel, GRB=GRB)
    return mock_gp, mock_scipy


def _capture_frames(generated_code: str, data: dict
                    ) -> tuple[dict, Optional[str]]:
    """Execute build_vectorized_model with synthetic data + mock gurobi.

    Returns (namespace, error). The namespace contains all locally-bound
    variables produced by the generated code (DataFrames, Series, indices).
    error is None on success, otherwise the exception message.
    """
    lines = generated_code.splitlines()
    func_lines: list[str] = []
    in_func = False
    for line in lines:
        if line.startswith('def build_vectorized_model'):
            in_func = True
            continue
        if in_func and line.startswith('def ') and 'build_vectorized' not in line:
            break
        if in_func:
            s = line.strip()
            if s.startswith('import ') or s.startswith('from '):
                continue
            if s.startswith('return ') or s == 'return':
                continue
            if 'setObjective' in s or '_get_values' in s:
                continue
            func_lines.append(line)

    if not func_lines:
        return {}, 'no build_vectorized_model body'

    import pandas as pd
    import numpy as np

    mock_gp, mock_scipy = _make_mock_gp()
    ns: dict = {
        'gp': mock_gp, 'data': data,
        'pd': pd, 'np': np, 'scipy': mock_scipy,
    }
    body = textwrap.dedent('\n'.join(func_lines))
    err: Optional[str] = None
    try:
        exec(body, ns)
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    return ns, err


# ===================================================================
# 3. Entity maps
# ===================================================================

def _build_pyomo_emap(spec: ModelSpec) -> dict[str, str]:
    """name → entity_key for Pyomo source code spans."""
    m: dict[str, str] = {}
    for n in spec.sets:
        m[n] = f'set-{n}'
    for n in spec.vars:
        m[n] = f'var-{n}'
    for n in spec.params:
        m[n] = f'param-{n}'
    return m


def _build_gurobi_emap(spec: ModelSpec, constr_name: str) -> dict[str, str]:
    """Generated-code identifier → entity_key for Gurobi code spans."""
    m: dict[str, str] = {}
    for vn in spec.vars:
        m[f'_var_{vn}'] = f'var-{vn}'
        m[f'_idx_{vn}'] = f'var-{vn}'
        m[f'_flat_{vn}'] = f'var-{vn}'
    for pn, pi in spec.params.items():
        dk = pi.data_key.lower()
        m[f's_{dk}'] = f'param-{pn}'
    cn = constr_name
    for pfx in ('_constr_', '_coo_', '_A_', '_A_sum_', '_A_neg_',
                '_idx_', '_m1_', '_fp_', '_dc_'):
        m[f'{pfx}{cn}'] = f'constr-{cn}'
    return m


# ===================================================================
# 4. Code panel HTML tagger
# ===================================================================

def _escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _tag_code(source: str, emap: dict[str, str]) -> str:
    """Wrap entity-matching identifiers in source with colored spans."""
    if not emap:
        return _escape(source)
    names = sorted(emap, key=len, reverse=True)
    pat = r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b'
    result: list[str] = []
    last = 0
    for match in re.finditer(pat, source):
        result.append(_escape(source[last:match.start()]))
        name = match.group(1)
        eid = emap[name]
        result.append(f'<span class="ent-{eid}">{_escape(name)}</span>')
        last = match.end()
    result.append(_escape(source[last:]))
    return ''.join(result)


# ===================================================================
# 5. Math rendering with entity wrappers (KaTeX \htmlClass)
# ===================================================================

_OP_TEX = {
    ast.LtE: r'\leq', ast.GtE: r'\geq', ast.Eq: '=',
    ast.Lt: '<', ast.Gt: '>',
}


def _tex_name(name: str) -> str:
    if len(name) == 1:
        return name
    safe = name.replace('_', r'\_')
    return r'\mathit{' + safe + r'}'


def _hcls(eid: str, tex: str) -> str:
    """Wrap tex in KaTeX \\htmlClass for entity eid."""
    return rf'\htmlClass{{ent-{eid}}}{{{tex}}}'


def _ast_to_math(node: ast.AST, assigns: dict, emap: Optional[dict] = None) -> str:
    """Render a Pyomo AST node as LaTeX, optionally tagging entities."""

    def R(n):
        return _ast_to_math(n, assigns, emap)

    if node is None:
        return '?'

    if isinstance(node, ast.Name):
        if node.id in assigns:
            return R(assigns[node.id])
        raw = node.id
        if '_' in raw:
            base = r'\mathit{' + raw.replace('_', r'\_') + r'}'
        else:
            base = raw
        if emap and raw in emap:
            return _hcls(emap[raw], base)
        return base

    if isinstance(node, ast.Constant):
        return str(node.value)

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == 'm':
            name = node.attr
            tex = _tex_name(name)
            if emap and name in emap:
                return _hcls(emap[name], tex)
            return tex
        return f'{R(node.value)}.{node.attr}'

    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Tuple):
            sub_args = ','.join(R(e) for e in sl.elts)
        else:
            sub_args = R(sl)

        # data['key'] → render as param-like name
        if isinstance(node.value, ast.Name) and node.value.id == 'data':
            inside = sl.value if isinstance(sl, ast.Constant) else None
            return _tex_name(str(inside)) if inside is not None else f'data[{sub_args}]'

        # m.Attr[...] — put subscript INSIDE the entity span
        if isinstance(node.value, ast.Attribute) and _is_m_attr(node.value):
            name = node.value.attr
            tex = _tex_name(name)
            inner = f'{tex}_{{{sub_args}}}'
            if emap and name in emap:
                return _hcls(emap[name], inner)
            return inner

        base_tex = R(node.value)
        return f'{base_tex}_{{{sub_args}}}'

    if isinstance(node, ast.BinOp):
        l, r = R(node.left), R(node.right)
        if isinstance(node.op, ast.Add):
            return f'{l} + {r}'
        if isinstance(node.op, ast.Sub):
            return f'{l} - {r}'
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)):
                return f'{node.left.value}\\,{r}'
            return fr'{l} \cdot {r}'
        if isinstance(node.op, ast.Div):
            return fr'\frac{{{l}}}{{{r}}}'
        return f'{l}?{r}'

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f'-{R(node.operand)}'
        return R(node.operand)

    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Name) and node.func.id == 'sum'
                and len(node.args) == 1):
            arg = node.args[0]
            if isinstance(arg, ast.GeneratorExp):
                elt_tex = R(arg.elt)
                bind_parts = []
                for g in arg.generators:
                    target_tex = _render_target(g.target, emap)
                    iter_tex = R(g.iter)
                    bind_parts.append(fr'{target_tex} \in {iter_tex}')
                return r'\sum_{' + r',\,'.join(bind_parts) + r'} ' + elt_tex
        fn = R(node.func)
        args = ','.join(R(a) for a in node.args)
        return f'{fn}({args})'

    if isinstance(node, ast.Compare):
        if len(node.ops) > 1:
            return R(node.left) + ''.join(
                f' {_OP_TEX.get(type(op), "?")} {R(comp)}'
                for op, comp in zip(node.ops, node.comparators)
            )
        l = R(node.left)
        op = _OP_TEX.get(type(node.ops[0]), '?')
        r = R(node.comparators[0])
        return fr'{l} \;{op}\; {r}'

    if isinstance(node, ast.Tuple):
        return f'({",".join(R(e) for e in node.elts)})'

    return f'\\text{{?{type(node).__name__}}}'


def _render_target(target, emap=None) -> str:
    if isinstance(target, ast.Name):
        raw = target.id
        if '_' in raw:
            return r'\mathit{' + raw.replace('_', r'\_') + r'}'
        return raw
    if isinstance(target, ast.Tuple):
        return f'({",".join(_render_target(e, emap) for e in target.elts)})'
    return '?'


def render_pyomo_math(c: ConstrSpec, spec: ModelSpec,
                      emap: Optional[dict] = None) -> tuple[str, Optional[str]]:
    if c.rule_body_node is None:
        return '', 'No rule body.'
    try:
        body = _ast_to_math(c.rule_body_node, c.rule_assigns, emap)
    except Exception as e:
        return '', str(e)
    if c.rule_args and c.index_sets:
        bind_parts = []
        for arg, sname in zip(c.rule_args, c.index_sets):
            sname_tex = _tex_name(sname)
            if emap and sname in emap:
                sname_tex = _hcls(emap[sname], sname_tex)
            bind_parts.append(fr'{arg} \in {sname_tex}')
        forall = r'\quad \forall\, ' + r',\,'.join(bind_parts)
        return body + forall, None
    return body, None


def render_obj_math(o: ObjSpec, emap: Optional[dict] = None) -> tuple[str, Optional[str]]:
    if o.rule_body_node is None:
        return '', 'No objective body.'
    try:
        body = _ast_to_math(o.rule_body_node, {}, emap)
    except Exception as e:
        return '', str(e)
    op = r'\min' if o.sense == 'MINIMIZE' else r'\max'
    return fr'{op}\quad {body}', None


# ===================================================================
# 6. Gurobi math — symbolic vectorized form from Pyomo AST
# ===================================================================

def _ent_var_tex(name: str, emap: dict) -> str:
    eid = emap.get(name, f'var-{name}')
    # Bold for single-char (clean), mathit for multi-char
    inner = fr'\mathbf{{{name}}}' if len(name) == 1 else fr'\mathit{{{name.replace("_", r"\_")}}}'
    return _hcls(eid, inner)


def _ent_param_tex(name: str, emap: dict) -> str:
    eid = emap.get(name, f'param-{name}')
    inner = fr'\mathbf{{{name}}}' if len(name) == 1 else fr'\mathit{{{name.replace("_", r"\_")}}}'
    return _hcls(eid, inner)


def _extract_sum_info(sum_node: ast.Call, assigns: dict,
                      emap: dict) -> Optional[tuple[str, str]]:
    """Return (coeff_tex, var_tex) from a sum(... for ...) node, or None."""
    gen = sum_node.args[0] if sum_node.args else None
    if not isinstance(gen, ast.GeneratorExp):
        return None
    elt = gen.elt

    # sum(m.var[i] for ...) → coeff=1
    if (isinstance(elt, ast.Subscript) and isinstance(elt.value, ast.Attribute)
            and _is_m_attr(elt.value)):
        vname = elt.value.attr
        return ('1', _ent_var_tex(vname, emap))

    # sum(coeff * m.var[i] for ...) or sum(m.var[i] * coeff for ...)
    if isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.Mult):
        l, r = elt.left, elt.right
        if (isinstance(r, ast.Subscript) and isinstance(r.value, ast.Attribute)
                and _is_m_attr(r.value)):
            var_node, coeff_node = r, l
        elif (isinstance(l, ast.Subscript) and isinstance(l.value, ast.Attribute)
              and _is_m_attr(l.value)):
            var_node, coeff_node = l, r
        else:
            return None

        vname = var_node.value.attr
        var_tex = _ent_var_tex(vname, emap)

        if isinstance(coeff_node, ast.Constant):
            coeff_tex = str(coeff_node.value)
        elif (isinstance(coeff_node, ast.Subscript)
              and isinstance(coeff_node.value, ast.Attribute)
              and _is_m_attr(coeff_node.value)):
            pname = coeff_node.value.attr
            p_eid = emap.get(pname, f'param-{pname}')
            sl = coeff_node.slice
            if isinstance(sl, ast.Tuple):
                sub_args = ','.join(_ast_to_math(e, assigns) for e in sl.elts)
            else:
                sub_args = _ast_to_math(sl, assigns)
            inner = f'{_tex_name(pname)}_{{{sub_args}}}'
            coeff_tex = _hcls(p_eid, inner)
        elif isinstance(coeff_node, ast.UnaryOp) and isinstance(coeff_node.op, ast.USub):
            coeff_tex = '-' + _ast_to_math(coeff_node.operand, assigns, emap)
        else:
            coeff_tex = _ast_to_math(coeff_node, assigns, emap)

        return (coeff_tex, var_tex)

    return None


def derive_gurobi_math(c: ConstrSpec, spec: ModelSpec,
                       emap: dict) -> tuple[str, Optional[str]]:
    """Produce symbolic vectorized (Gurobi/matrix) form from Pyomo ConstrSpec."""
    body = c.rule_body_node
    if body is None or not isinstance(body, ast.Compare):
        return '', 'No comparable rule body.'

    op_tex = _OP_TEX.get(type(body.ops[0]), '?') if body.ops else '?'

    def _res(node):
        if isinstance(node, ast.Name) and node.id in c.rule_assigns:
            return c.rule_assigns[node.id]
        return node

    lhs = _res(body.left)
    rhs = _res(body.comparators[0]) if body.comparators else None

    def _rhs_tex():
        if rhs is None:
            return '?'
        rn = _res(rhs)
        if (isinstance(rn, ast.Subscript) and isinstance(rn.value, ast.Attribute)
                and _is_m_attr(rn.value)):
            name = rn.value.attr
            if name in spec.vars:
                return _ent_var_tex(name, emap)
            return _ent_param_tex(name, emap)
        dk = _is_data_subscript(rn)
        if dk:
            return _tex_name(dk)
        return _ast_to_math(rn, c.rule_assigns, emap)

    # --- Case A: direct subscript on LHS ---
    if (isinstance(lhs, ast.Subscript) and isinstance(lhs.value, ast.Attribute)
            and _is_m_attr(lhs.value)):
        lname = lhs.value.attr

        # Sub-case: LHS is a variable, RHS is a sum (component-balance style)
        if (lname in spec.vars and rhs is not None
                and isinstance(_res(rhs), ast.Call)
                and isinstance(_res(rhs).func, ast.Name)
                and _res(rhs).func.id == 'sum'):
            info = _extract_sum_info(_res(rhs), c.rule_assigns, emap)
            if info:
                coeff_tex, sum_var_tex = info
                lv = _ent_var_tex(lname, emap)
                return (
                    fr'\mathbf{{A}} \cdot {sum_var_tex} - \mathbf{{I}} \cdot {lv} = \mathbf{{0}}'
                    fr'\quad (\mathbf{{A}}_{{ij}} = {coeff_tex})',
                    None
                )

        # Sub-case: LHS is a variable or param; identity pattern
        if lname in spec.vars:
            vt = _ent_var_tex(lname, emap)
            rt = _rhs_tex()
            return fr'\mathbf{{I}} \cdot {vt} \;{op_tex}\; {rt}', None

        # LHS is a param (unusual) — fall through

    # --- Case B: single sum on LHS ---
    if (isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name)
            and lhs.func.id == 'sum'):
        info = _extract_sum_info(lhs, c.rule_assigns, emap)
        if info:
            coeff_tex, sum_var_tex = info
            # Check if RHS is a direct variable (LHS equality)
            if rhs is not None:
                rn = _res(rhs)
                if (isinstance(rn, ast.Subscript) and isinstance(rn.value, ast.Attribute)
                        and _is_m_attr(rn.value) and rn.value.attr in spec.vars):
                    rv = _ent_var_tex(rn.value.attr, emap)
                    return (
                        fr'\mathbf{{A}} \cdot {sum_var_tex} - \mathbf{{I}} \cdot {rv} = \mathbf{{0}}'
                        fr'\quad (\mathbf{{A}}_{{ij}} = {coeff_tex})',
                        None
                    )
            rt = _rhs_tex()
            return (
                fr'\mathbf{{A}} \cdot {sum_var_tex} \;{op_tex}\; {rt}'
                fr'\quad (\mathbf{{A}}_{{ij}} = {coeff_tex})',
                None
            )

    # --- Case C: flow balance (sum_out - sum_in op rhs) ---
    if isinstance(lhs, ast.BinOp) and isinstance(lhs.op, ast.Sub):
        out_n = _res(lhs.left)
        in_n = _res(lhs.right)
        var_name = None
        for n in (out_n, in_n):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'sum'):
                gen = n.args[0] if n.args else None
                if isinstance(gen, ast.GeneratorExp):
                    elt = gen.elt
                    if (isinstance(elt, ast.Subscript)
                            and isinstance(elt.value, ast.Attribute)
                            and _is_m_attr(elt.value)):
                        var_name = elt.value.attr
                        break
        if var_name:
            vt = _ent_var_tex(var_name, emap)
            rt = _rhs_tex()
            return (
                fr'(\mathbf{{A}}_+ - \mathbf{{A}}_-) \cdot {vt} \;{op_tex}\; {rt}'
                r'\quad (\mathbf{A}_+ \text{ out},\ \mathbf{A}_- \text{ in})',
                None
            )

    # --- Case D: LHS is a BinOp add (multi-term, e.g. inter-sum) ---
    if isinstance(lhs, ast.BinOp) and isinstance(lhs.op, ast.Add):
        parts = []
        for term in (lhs.left, lhs.right):
            tn = _res(term)
            if (isinstance(tn, ast.Call) and isinstance(tn.func, ast.Name)
                    and tn.func.id == 'sum'):
                info = _extract_sum_info(tn, c.rule_assigns, emap)
                if info:
                    coeff_tex, sum_var_tex = info
                    parts.append(fr'\mathbf{{A}} \cdot {sum_var_tex}')
        if parts:
            rt = _rhs_tex()
            return fr'{" + ".join(parts)} \;{op_tex}\; {rt}', None

    # --- Fallback: render as Pyomo math ---
    tex, err = render_pyomo_math(c, spec, emap)
    if err:
        return '', f'Cannot derive: {err}'
    return tex, None


def derive_gurobi_obj_math(o: ObjSpec, spec: ModelSpec,
                           emap: dict) -> tuple[str, Optional[str]]:
    """Derive Gurobi math for objective."""
    body = o.rule_body_node
    if body is None:
        return '', 'No objective body.'

    sense_tex = r'\min' if o.sense == 'MINIMIZE' else r'\max'

    def _term_to_vec(node) -> Optional[str]:
        """Convert a sum() node to c^T·x notation."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'sum'):
            return None
        info = _extract_sum_info(node, {}, emap)
        if not info:
            return None
        coeff_tex, var_tex = info
        if coeff_tex == '1':
            return fr'\mathbf{{c}}^\top \cdot {var_tex}'
        return fr'{coeff_tex}^\top \cdot {var_tex}'

    # Multi-term: term + term or term - term
    if isinstance(body, ast.BinOp):
        lt = _term_to_vec(body.left)
        rt = _term_to_vec(body.right)
        if lt and rt:
            op = '+' if isinstance(body.op, ast.Add) else '-'
            return fr'{sense_tex}\quad {lt} {op} {rt}', None

    vt = _term_to_vec(body)
    if vt:
        return fr'{sense_tex}\quad {vt}', None

    # Fallback
    tex, err = render_obj_math(o, emap)
    if err:
        return '', err
    return fr'{sense_tex}\quad {tex}', None


# ===================================================================
# 7. Walkthrough builder — extracts step data per constraint
# ===================================================================

@dataclass
class WalkthroughBlock:
    """One block of variables-to-constraint contribution.
    A single constraint may have one (P1/P3/P5) or two (P4/intra-sum) blocks.
    """
    var_name: str                                  # 'build', 'buy_comp', 'x', ...
    coeff_label: Optional[str]                     # 'BOM[p,c]' or '−1' or '1'
    coeff_kind: str                                # 'param', 'unit', 'neg_unit', 'flow', 'unknown'
    var_frame: Any                                 # _flat_<var> DataFrame
    coo_frame: Any                                 # merged frame backing this block
    value_col: Optional[str]                       # value col name in coo_frame
    implicit_value: Optional[float]                # value when no value column
    row_col: Optional[str] = '_row'                # name of the row-index column in coo_frame
    col_col: Optional[str] = '_col'                # name of the col-index column in coo_frame
    sign: str = '+'


@dataclass
class Walkthrough:
    constraint_name: str
    constr_frame: Any                              # _constr_<name> DataFrame
    constr_axes: list[str]                         # column names of the constraint index (excluding _row)
    blocks: list[WalkthroughBlock]                 # one per variable in the LHS
    matrix: Any                                    # dense 2D array, rows × total_cols
    matrix_col_offsets: list[int]                  # where each block starts in the concatenated matrix
    error_step1: Optional[str] = None
    error_step2: Optional[str] = None
    error_step3: Optional[str] = None
    error_step4: Optional[str] = None
    name_mapping: list[tuple[str, str]] = field(default_factory=list)


# Block parsing — discover (matrix_var, var_name, sign) tuples from add* lines,
# then for each matrix_var parse the `_A_<matrix_var> = csr_matrix(...)` line
# to discover the underlying frame name and row/col/value column references.


def _parse_constraint_blocks(generated_code: str, constr_name: str
                             ) -> list[tuple[str, str, str]]:
    """Return (matrix_var, var_name, sign) for every block referenced by
    the add*/addMConstr call for this constraint.

    matrix_var is the suffix on `_A_`. It may contain constr_name in any
    position (e.g. `capacity_constr`, `sum_component_constr`, `cap_constr_t0`).
    """
    cn = re.escape(constr_name)
    # Any `_A_xxx` where xxx contains constr_name as a token
    A_pat = rf"_A_((?:\w+_)?{cn}(?:_\w+)?)"

    # addMConstr: single block
    addm_pat = re.compile(
        rf"m\.addMConstr\(\s*{A_pat}\s*,\s*_var_(\w+)\s*,\s*gp\.GRB\."
    )
    for m in addm_pat.finditer(generated_code):
        return [(m.group(1), m.group(2), '+')]

    # addConstr: scan inside the parentheses for all _A_xxx @ _var_yyy terms
    addc_pat = re.compile(
        rf"m\.addConstr\(\s*\((.*?)\)\s*[=<>]", re.DOTALL,
    )
    body_match = addc_pat.search(generated_code[generated_code.find(
        f"m._mconstr['{constr_name}']"):]) if (
        f"m._mconstr['{constr_name}']" in generated_code) else None
    if not body_match:
        # Try unanchored on entire code
        body_match = addc_pat.search(generated_code)
    if not body_match:
        return []
    body = body_match.group(1)

    term_pat = re.compile(rf"([+\-]?)\s*{A_pat}\s*@\s*_var_(\w+)")
    blocks: list[tuple[str, str, str]] = []
    for tm in term_pat.finditer(body):
        sign = tm.group(1) or '+'
        if sign == '':
            sign = '+'
        blocks.append((tm.group(2), tm.group(3), sign))
    return blocks


@dataclass
class _MatrixDef:
    frame_name: Optional[str]   # e.g. '_coo_xxx', '_mi_xxx', '_dc_xxx'
    row_col: Optional[str]      # e.g. '_row', or None for scalar (np.zeros)
    col_col: Optional[str]      # e.g. '_col', '_col_t0'
    value_col: Optional[str]    # column name in the frame, or None
    implicit_value: Optional[float]  # 1.0 / -1.0 / 0.5 etc., or None
    n_rows_scalar: Optional[int]    # for scalar constraints, the row count (typically 1)


def _split_top_commas(s: str) -> list[str]:
    """Split a string on top-level commas (ignoring nested parens/brackets)."""
    parts: list[str] = []
    depth = 0
    cur = []
    for ch in s:
        if ch in '([{':
            depth += 1
            cur.append(ch)
        elif ch in ')]}':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


def _balanced_call(s: str, start: int) -> Optional[str]:
    """Given s[start] == '(', return contents between that '(' and matching ')'."""
    if start >= len(s) or s[start] != '(':
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return s[start + 1: i]
    return None


def _parse_matrix_def(generated_code: str, matrix_var: str
                      ) -> Optional[_MatrixDef]:
    """Parse `_A_<matrix_var> = scipy.sparse.csr_matrix((data, (rows, cols)), shape=...)`.

    Generic, paren-aware extraction. Returns _MatrixDef or None.
    """
    needle = f'_A_{matrix_var} = scipy.sparse.csr_matrix('
    idx = generated_code.find(needle)
    if idx < 0:
        return None
    open_idx = idx + len(needle) - 1  # the '(' of csr_matrix(
    body = _balanced_call(generated_code, open_idx)
    if body is None:
        return None
    # body looks like: "(DATA, (ROWS, COLS)), shape=..."
    # First arg is the outer-paren tuple, second is shape.
    if not body.lstrip().startswith('('):
        return None
    inner_open = body.find('(')
    inner = _balanced_call(body, inner_open)
    if inner is None:
        return None
    parts = _split_top_commas(inner)
    if len(parts) != 2:
        return None
    data_arg = parts[0]
    coord_arg = parts[1]
    if not coord_arg.startswith('('):
        return None
    coords = _balanced_call(coord_arg, 0)
    if coords is None:
        return None
    coord_parts = _split_top_commas(coords)
    if len(coord_parts) != 2:
        return None
    rowref, colref = coord_parts[0], coord_parts[1]

    cell_pat = re.compile(r"(\w+)\[\s*'(\w+)'\s*\]\.values")
    rm = cell_pat.match(rowref)
    cm = cell_pat.match(colref)

    frame_name: Optional[str] = None
    row_col: Optional[str] = None
    n_rows_scalar: Optional[int] = None

    if rm:
        frame_name = rm.group(1)
        row_col = rm.group(2)
    elif 'np.zeros' in rowref or rowref.startswith('np.full'):
        n_rows_scalar = 1

    col_col: Optional[str] = None
    if cm:
        if frame_name is None:
            frame_name = cm.group(1)
        col_col = cm.group(2)

    value_col: Optional[str] = None
    implicit: Optional[float] = None
    if data_arg.startswith('-np.ones('):
        implicit = -1.0
    elif data_arg.startswith('np.ones('):
        implicit = 1.0
    elif data_arg.startswith('-np.full('):
        fm = re.search(r'np\.full\([^,]+,\s*([\-\d.+e]+)\s*\)', data_arg)
        implicit = -float(fm.group(1)) if fm else -1.0
    elif data_arg.startswith('np.full('):
        fm = re.search(r'np\.full\([^,]+,\s*([\-\d.+e]+)\s*\)', data_arg)
        implicit = float(fm.group(1)) if fm else 1.0
    else:
        vm = re.search(r"\[\s*'(?!_row|_col)(\w+)'\s*\]\.values", data_arg)
        if vm:
            value_col = vm.group(1)

    return _MatrixDef(
        frame_name=frame_name, row_col=row_col, col_col=col_col,
        value_col=value_col, implicit_value=implicit,
        n_rows_scalar=n_rows_scalar,
    )


def _extract_constraint_axes(constr_frame, generated_code: str,
                             constr_name: str) -> list[str]:
    """Get the constraint's index axis names (columns excluding _row)."""
    if constr_frame is None or not hasattr(constr_frame, 'columns'):
        return []
    return [c for c in constr_frame.columns if c != '_row']


def _build_matrix(blocks: list[WalkthroughBlock], scalar_rows: int
                  ) -> tuple[Any, list[int], int]:
    """Assemble the dense matrix [A_block1 | A_block2 | ...] from block COO frames."""
    import numpy as np

    if not blocks:
        return None, [], 0

    # Determine row count: scalar override > max _row in any block coo
    n_rows = scalar_rows
    for b in blocks:
        if b.coo_frame is None or b.row_col is None:
            continue
        if b.row_col not in getattr(b.coo_frame, 'columns', []):
            continue
        if len(b.coo_frame) > 0:
            n_rows = max(n_rows, int(b.coo_frame[b.row_col].max()) + 1)

    # Per-block column count from the var_frame
    col_counts = []
    for b in blocks:
        col_counts.append(len(b.var_frame) if b.var_frame is not None else 0)

    offsets, cum = [], 0
    for cc in col_counts:
        offsets.append(cum)
        cum += cc
    total_cols = cum

    if n_rows == 0 or total_cols == 0:
        return None, offsets, total_cols

    M = np.zeros((n_rows, total_cols), dtype=float)

    for bi, b in enumerate(blocks):
        coo = b.coo_frame
        if coo is None or b.col_col is None or b.col_col not in coo.columns:
            continue
        cols = coo[b.col_col].values
        if b.row_col is not None and b.row_col in coo.columns:
            rows = coo[b.row_col].values
        else:
            # Scalar row constraint: all rows are 0
            rows = [0] * len(coo)

        sign = -1.0 if b.sign == '-' else 1.0

        if b.value_col is not None and b.value_col in coo.columns:
            vals = coo[b.value_col].values.astype(float)
        elif '_val' in coo.columns:
            vals = coo['_val'].values.astype(float)
        elif b.implicit_value is not None:
            vals = np.full(len(coo), float(b.implicit_value))
        else:
            vals = np.ones(len(coo))

        col_offset = offsets[bi]
        for r, c, v in zip(rows, cols, vals):
            M[int(r), int(c) + col_offset] += sign * float(v)

    return M, offsets, total_cols


def build_walkthrough(c: ConstrSpec, spec: ModelSpec,
                      generated_code: str, ns: dict
                      ) -> Walkthrough:
    """Construct a Walkthrough for the given constraint from captured frames."""
    wt = Walkthrough(
        constraint_name=c.name,
        constr_frame=ns.get(f'_constr_{c.name}'),
        constr_axes=[],
        blocks=[],
        matrix=None,
        matrix_col_offsets=[],
    )

    wt.constr_axes = _extract_constraint_axes(wt.constr_frame, generated_code, c.name)

    block_specs = _parse_constraint_blocks(generated_code, c.name)
    if not block_specs:
        wt.error_step1 = (
            f'No matrix block found for constraint {c.name} — '
            'could not locate addMConstr/addConstr line')
        return wt

    blocks: list[WalkthroughBlock] = []
    scalar_rows = 0
    for matrix_var, var_name, sign in block_specs:
        info = _parse_matrix_def(generated_code, matrix_var)
        if info is None:
            wt.error_step3 = (wt.error_step3 or '') + (
                f' could not parse _A_{matrix_var} = csr_matrix(...);')
            continue

        var_frame = ns.get(f'_flat_{var_name}')
        coo_frame = ns.get(info.frame_name) if info.frame_name else None

        if info.n_rows_scalar is not None:
            scalar_rows = max(scalar_rows, info.n_rows_scalar)
            # For scalar constraints, _constr_<cn> may not exist
            if wt.constr_frame is None:
                # Build a synthetic 1-row frame
                import pandas as _pd
                wt.constr_frame = _pd.DataFrame({'_row': [0], 'row': ['(scalar)']})
                wt.constr_axes = ['row']

        if wt.constr_frame is None:
            wt.error_step2 = (
                f'_constr_{c.name} was not produced — '
                'generated code may not match expected pattern')

        # Determine coefficient label & kind
        if info.value_col is not None:
            coeff_kind = 'param'
            coeff_label = info.value_col

            # First try param-name reverse lookup (e.g. 'bom' → BOM, 'routing' → Routing)
            pname = None
            for pn, ps in spec.params.items():
                if (ps.data_key.lower() == info.value_col
                        or pn.lower() == info.value_col):
                    pname = pn
                    break
            if pname:
                idx = ','.join(spec.params[pname].index_sets)
                coeff_label = f'{pname}[{idx}]' if idx else pname

            # Special: value_col == '_val' — inspect actual values + source code
            elif info.value_col == '_val' and coo_frame is not None:
                try:
                    uniq = set(float(v) for v in coo_frame['_val'].values)
                except Exception:
                    uniq = set()
                if uniq and uniq.issubset({1.0, -1.0}) and len(uniq) > 1:
                    coeff_kind = 'flow'
                    coeff_label = '±1'
                elif uniq == {1.0}:
                    coeff_kind = 'unit'
                    coeff_label = '1'
                elif uniq == {-1.0}:
                    coeff_kind = 'neg_unit'
                    coeff_label = '−1'
                else:
                    # Look for `.map(s_<param>)` source in generated code
                    cn = re.escape(c.name)
                    map_pat = re.compile(
                        rf"_coo_{cn}\s*=\s*[^\n]*\.map\(\s*s_(\w+)\s*\)")
                    mm = map_pat.search(generated_code)
                    if mm:
                        plow = mm.group(1)
                        for pn, ps in spec.params.items():
                            if pn.lower() == plow or ps.data_key.lower() == plow:
                                idx = ','.join(ps.index_sets)
                                coeff_label = f'{pn}[{idx}]' if idx else pn
                                break
        elif info.implicit_value is not None and info.implicit_value > 0:
            coeff_kind = 'unit'
            coeff_label = '1' if info.implicit_value == 1.0 else f'{info.implicit_value:g}'
        elif info.implicit_value is not None and info.implicit_value < 0:
            coeff_kind = 'neg_unit'
            coeff_label = '−1' if info.implicit_value == -1.0 else f'{info.implicit_value:g}'
        elif coo_frame is not None and '_val' in getattr(coo_frame, 'columns', []):
            coeff_kind = 'flow'
            coeff_label = '±1'
        else:
            coeff_kind = 'unknown'
            coeff_label = '?'

        if var_frame is None:
            wt.error_step1 = (wt.error_step1 or '') + (
                f' missing _flat_{var_name};')
        if coo_frame is None and info.frame_name:
            wt.error_step3 = (wt.error_step3 or '') + (
                f' missing {info.frame_name};')

        blocks.append(WalkthroughBlock(
            var_name=var_name,
            coeff_label=coeff_label,
            coeff_kind=coeff_kind,
            var_frame=var_frame,
            coo_frame=coo_frame,
            value_col=info.value_col,
            implicit_value=info.implicit_value,
            row_col=info.row_col,
            col_col=info.col_col,
            sign=sign,
        ))

    wt.blocks = blocks

    try:
        M, offsets, total = _build_matrix(blocks, scalar_rows)
        wt.matrix = M
        wt.matrix_col_offsets = offsets
        if M is None:
            wt.error_step4 = 'Could not assemble matrix — block frames empty or malformed'
    except Exception as e:
        wt.error_step4 = f'Matrix assembly failed: {type(e).__name__}: {e}'

    return wt


# ===================================================================
# 7b. Walkthrough HTML renderer
# ===================================================================

def _df_pretty_row(row, cols, value_col=None) -> dict:
    out = {}
    for c in cols:
        v = row[c]
        if isinstance(v, float):
            out[c] = f'{v:.4g}'
        else:
            out[c] = str(v)
    return out


def _render_step1(wt: Walkthrough) -> str:
    """Variable numbering tables — one per variable involved."""
    if wt.error_step1:
        return _render_step_error(1, 'Number every variable', wt.error_step1)
    parts = ['<div class="step">',
             '<div class="step-head"><span class="step-num">Step 1</span> '
             '<span class="step-title">Number every variable</span>'
             '<span class="step-sub">Each variable instance gets a column number.</span></div>']
    parts.append('<div class="step-body step1-body">')
    seen_vars = set()
    for b in wt.blocks:
        if b.var_name in seen_vars:
            continue
        seen_vars.add(b.var_name)
        parts.append(_render_var_table(b))
    parts.append('</div></div>')
    return ''.join(parts)


def _render_var_table(b: WalkthroughBlock) -> str:
    if b.var_frame is None or not hasattr(b.var_frame, 'columns'):
        return (f'<div class="step-block-err">⚠ Could not enumerate '
                f'variable {_escape(b.var_name)}: _flat_{b.var_name} missing</div>')
    df = b.var_frame
    cols = [c for c in df.columns if c != '_col']
    parts = [f'<div class="var-table-wrap">',
             f'<div class="step-label">Variable: '
             f'<span class="ent-var-{b.var_name} ent-token">{_escape(b.var_name)}'
             f'<span class="subscript">[{",".join(cols)}]</span></span></div>',
             '<table class="wt-table"><thead><tr>']
    for c in cols:
        parts.append(f'<th class="wt-th-axis">{_escape(c)}</th>')
    parts.append(f'<th class="wt-th-int">col #</th></tr></thead><tbody>')
    for _, row in df.iterrows():
        parts.append('<tr>')
        for c in cols:
            v = row[c]
            parts.append(f'<td class="wt-td-axis">{_escape(str(v))}</td>')
        parts.append(f'<td class="wt-td-int ent-var-{b.var_name}">'
                     f'{int(row["_col"])}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _render_step2(wt: Walkthrough) -> str:
    """Constraint row numbering table."""
    if wt.error_step2:
        return _render_step_error(2, 'Number every constraint row', wt.error_step2)
    if wt.constr_frame is None:
        return _render_step_error(2, 'Number every constraint row',
                                  'Constraint row table not captured.')
    df = wt.constr_frame
    cols = [c for c in df.columns if c != '_row']
    parts = ['<div class="step">',
             '<div class="step-head"><span class="step-num">Step 2</span> '
             '<span class="step-title">Number every constraint row</span>'
             f'<span class="step-sub">Each instance of <span class="ent-constr-{wt.constraint_name} ent-token">'
             f'{_escape(wt.constraint_name)}</span> gets a row number.</span></div>',
             '<div class="step-body">',
             '<table class="wt-table"><thead><tr>']
    for c in cols:
        parts.append(f'<th class="wt-th-axis">{_escape(c)}</th>')
    parts.append(f'<th class="wt-th-int">row #</th></tr></thead><tbody>')
    for _, row in df.iterrows():
        parts.append('<tr>')
        for c in cols:
            parts.append(f'<td class="wt-td-axis">{_escape(str(row[c]))}</td>')
        parts.append(f'<td class="wt-td-int ent-constr-{wt.constraint_name}">'
                     f'{int(row["_row"])}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div></div>')
    return ''.join(parts)


def _render_step3(wt: Walkthrough) -> str:
    """Merge tables — one per block."""
    if wt.error_step3:
        return _render_step_error(3, 'Match variables to constraint rows',
                                  wt.error_step3)
    parts = ['<div class="step">',
             '<div class="step-head"><span class="step-num">Step 3</span> '
             '<span class="step-title">Match variables to constraint rows</span>'
             '<span class="step-sub">Join on shared index columns. '
             'Each row says: place this coefficient at (row, col) of the matrix.</span></div>',
             '<div class="step-body step3-body">']
    for bi, b in enumerate(wt.blocks):
        parts.append(_render_merge_table(wt, b, bi))
    parts.append('</div></div>')
    return ''.join(parts)


def _render_merge_table(wt: Walkthrough, b: WalkthroughBlock, bi: int) -> str:
    if b.coo_frame is None or not hasattr(b.coo_frame, 'columns'):
        return (f'<div class="step-block-err">⚠ Could not render block {bi+1} for '
                f'<span class="ent-var-{b.var_name}">{_escape(b.var_name)}</span>: '
                f'merge frame missing.</div>')

    df = b.coo_frame
    has_val = '_val' in df.columns or (b.value_col is not None and b.value_col in df.columns)
    val_label = b.coeff_label or 'value'

    # Decide displayed columns:
    # show join keys (non-_), then _col, then _row, then value
    join_keys = [c for c in df.columns if c not in ('_row', '_col', '_val')
                 and c != b.value_col]
    has_col = '_col' in df.columns
    has_row = '_row' in df.columns

    parts = [f'<div class="merge-table-wrap">']
    parts.append(
        f'<div class="step-label">Block {bi+1}: '
        f'coefficient <span class="ent-{"param" if b.coeff_kind=="param" else "var"}-{b.var_name} ent-token">'
        f'{_escape(val_label)}</span> on variable '
        f'<span class="ent-var-{b.var_name} ent-token">{_escape(b.var_name)}</span></div>')
    parts.append('<table class="wt-table"><thead><tr>')
    for c in join_keys:
        parts.append(f'<th class="wt-th-axis">{_escape(c)}</th>')
    if has_col:
        parts.append(f'<th class="wt-th-int">col #</th>')
    if has_row:
        parts.append(f'<th class="wt-th-int">row #</th>')
    parts.append('<th class="wt-th-val">value</th>')
    parts.append('</tr></thead><tbody>')

    for _, row in df.iterrows():
        parts.append('<tr>')
        for c in join_keys:
            parts.append(f'<td class="wt-td-axis">{_escape(str(row[c]))}</td>')
        if has_col:
            parts.append(f'<td class="wt-td-int ent-var-{b.var_name}">{int(row["_col"])}</td>')
        if has_row:
            parts.append(f'<td class="wt-td-int ent-constr-{wt.constraint_name}">{int(row["_row"])}</td>')
        # Value cell
        if b.value_col is not None and b.value_col in df.columns:
            v = row[b.value_col]
            vs = f'{v:.4g}' if isinstance(v, float) else str(v)
            parts.append(f'<td class="wt-td-val ent-param-{b.value_col}">{_escape(vs)}</td>')
        elif '_val' in df.columns:
            v = float(row['_val'])
            parts.append(f'<td class="wt-td-val">{v:.4g}</td>')
        else:
            iv = b.implicit_value if b.implicit_value is not None else 1.0
            parts.append(f'<td class="wt-td-val">{iv:g}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _render_step4(wt: Walkthrough) -> str:
    """Resulting matrix Ax (≤/=/≥) b — visual matrix with row+col labels."""
    if wt.error_step4 or wt.matrix is None:
        msg = wt.error_step4 or 'Matrix not assembled.'
        return _render_step_error(4, 'Scatter into the matrix', msg)

    M = wt.matrix
    n_rows, n_cols = M.shape

    # Build column labels per block
    col_labels: list[tuple[str, str]] = []  # (var_name, subscript)
    block_boundaries = list(wt.matrix_col_offsets) + [n_cols]
    for bi, b in enumerate(wt.blocks):
        start = wt.matrix_col_offsets[bi]
        end = block_boundaries[bi + 1]
        if b.var_frame is None:
            for j in range(start, end):
                col_labels.append((b.var_name, str(j)))
            continue
        axes = [c for c in b.var_frame.columns if c != '_col']
        for _, row in b.var_frame.iterrows():
            sub = ','.join(str(row[a]) for a in axes)
            col_labels.append((b.var_name, sub))

    # Row labels from the constraint frame
    row_labels: list[str] = []
    if wt.constr_frame is not None and hasattr(wt.constr_frame, 'columns'):
        axes = [c for c in wt.constr_frame.columns if c != '_row']
        for _, row in wt.constr_frame.iterrows():
            row_labels.append(','.join(str(row[a]) for a in axes))
    else:
        row_labels = [str(i) for i in range(n_rows)]

    parts = ['<div class="step">',
             '<div class="step-head"><span class="step-num">Step 4</span> '
             '<span class="step-title">Scatter into the matrix</span>'
             '<span class="step-sub">Each (row, col, value) from Step 3 lands at its position. '
             f'Result: <code>A · x</code> for <span class="ent-constr-{wt.constraint_name} ent-token">'
             f'{_escape(wt.constraint_name)}</span>.</span></div>',
             '<div class="step-body step4-body">']
    parts.append('<table class="wt-matrix"><thead>')

    # Top header row 1: variable name spans
    parts.append('<tr><th class="mtx-corner" colspan="2"></th>')
    seen: list[tuple[str, int]] = []
    for vn, _ in col_labels:
        if seen and seen[-1][0] == vn:
            seen[-1] = (vn, seen[-1][1] + 1)
        else:
            seen.append((vn, 1))
    for vn, span in seen:
        parts.append(
            f'<th class="mtx-var-head ent-var-{vn} ent-token" colspan="{span}">'
            f'{_escape(vn)}</th>')
    parts.append('</tr>')

    # Top header row 2: column subscripts
    parts.append('<tr><th class="mtx-corner"></th><th class="mtx-corner">col #</th>')
    for j, (vn, sub) in enumerate(col_labels):
        parts.append(
            f'<th class="mtx-col-sub ent-var-{vn}"><span class="mtx-sub">{_escape(sub)}</span>'
            f'<span class="mtx-colnum">{j}</span></th>')
    parts.append('</tr></thead><tbody>')

    for i, rlabel in enumerate(row_labels):
        parts.append('<tr>')
        parts.append(
            f'<td class="mtx-row-sub ent-constr-{wt.constraint_name}">{_escape(rlabel)}</td>')
        parts.append(f'<td class="mtx-rownum">{i}</td>')
        for j in range(n_cols):
            v = M[i, j]
            if v == 0:
                parts.append('<td class="mtx-zero">·</td>')
            else:
                vs = f'{v:.4g}' if v != int(v) else str(int(v))
                cls = 'mtx-pos' if v > 0 else 'mtx-neg'
                parts.append(f'<td class="mtx-cell {cls}">{_escape(vs)}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div></div>')
    return ''.join(parts)


def _render_step_error(num: int, title: str, msg: str) -> str:
    return (f'<div class="step step-failed">'
            f'<div class="step-head"><span class="step-num">Step {num}</span> '
            f'<span class="step-title">{_escape(title)}</span></div>'
            f'<div class="step-body"><div class="step-err">'
            f'⚠ Could not render Step {num} — {_escape(msg)}'
            f'</div></div></div>')


def _render_walkthrough(wt: Walkthrough) -> str:
    return ('<div class="walkthrough">'
            '<div class="walkthrough-title">Walkthrough — '
            'how the constraint becomes a matrix</div>'
            + _render_step1(wt)
            + _render_step2(wt)
            + _render_step3(wt)
            + _render_step4(wt)
            + '</div>')


# ===================================================================
# 7c. Card builder
# ===================================================================

@dataclass
class Card:
    name: str
    kind: str                              # 'constraint' or 'objective'
    pyomo_code_html: str
    pyomo_math_tex: str
    pyomo_math_err: Optional[str]
    gurobi_math_tex: str
    gurobi_math_err: Optional[str]
    gurobi_code_html: str
    walkthrough_html: Optional[str] = None
    name_mapping_html: Optional[str] = None


def _render_gurobi_panel(raw_code: str, emap: dict) -> str:
    """Render Gurobi code slice with labeled micro-section headers."""
    if not raw_code.strip():
        return '<div class="math-err">⚠ No Gurobi code slice found.</div>'

    lines = raw_code.splitlines()
    groups: list[tuple[str, list[str]]] = []
    cur_sec: Optional[str] = None
    cur_lines: list[str] = []

    for line in lines:
        sec = _gurobi_section(line)
        if sec == 'blank':
            if cur_lines:
                cur_lines.append(line)
            continue
        if sec != cur_sec:
            if cur_sec is not None and cur_lines:
                groups.append((cur_sec, list(cur_lines)))
            cur_sec = sec
            cur_lines = [line]
        else:
            cur_lines.append(line)

    if cur_sec is not None and cur_lines:
        groups.append((cur_sec, cur_lines))

    if not groups:
        return f'<pre class="gsec-pre">{_tag_code(raw_code, emap)}</pre>'

    parts: list[str] = []
    for section, sec_lines in groups:
        label = _SECTION_LABELS.get(section, section)
        tagged = _tag_code('\n'.join(sec_lines), emap)
        parts.append(
            f'<div class="gsec-header">{_escape(label)}</div>'
            f'<pre class="gsec-pre">{tagged}</pre>'
        )
    return '\n'.join(parts)


def _build_name_mapping(spec: ModelSpec) -> Optional[str]:
    """Show name mapping rows ONLY when the Pyomo source name differs from
    the data key or the generated-code lowercased identifier.
    """
    rows: list[str] = []
    for pname, pspec in spec.params.items():
        # Pyomo name vs data key
        if pname != pspec.data_key:
            rows.append(
                f'<span class="nm-row">'
                f'<span class="ent-param-{pname} ent-token">{_escape(pname)}</span>'
                f' <span class="nm-arrow">↔</span> '
                f'<span class="nm-data">data[\'{_escape(pspec.data_key)}\']</span>'
                f' <span class="nm-arrow">↔</span> '
                f'<span class="nm-code">s_{pname.lower()}</span>'
                f'</span>')
    for sname, sspec in spec.sets.items():
        if sname != sspec.data_key:
            rows.append(
                f'<span class="nm-row">'
                f'<span class="ent-set-{sname} ent-token">{_escape(sname)}</span>'
                f' <span class="nm-arrow">↔</span> '
                f'<span class="nm-data">data[\'{_escape(sspec.data_key)}\']</span>'
                f'</span>')
    if not rows:
        return None
    return ('<div class="name-mapping">'
            '<span class="nm-label">Name mapping:</span> '
            + ' &nbsp;'.join(rows) + '</div>')


def _build_cards(spec: ModelSpec, generated_code: str,
                 build_pyomo_fn=None) -> list[Card]:
    pyomo_emap = _build_pyomo_emap(spec)
    slices = extract_gurobi_slices(generated_code)

    # Synthesize data and capture frames once per spec
    ns: dict = {}
    capture_err: Optional[str] = None
    if build_pyomo_fn is not None:
        try:
            data = _build_synthetic_data(spec, build_pyomo_fn)
            if data is None:
                capture_err = 'Failed to synthesize input data from spec.'
            else:
                ns, capture_err = _capture_frames(generated_code, data)
        except Exception as e:
            capture_err = f'{type(e).__name__}: {e}'

    name_mapping_html = _build_name_mapping(spec)

    cards: list[Card] = []

    for c in spec.constrs:
        gurobi_emap = _build_gurobi_emap(spec, c.name)
        constr_emap = dict(pyomo_emap)
        constr_emap[c.name] = f'constr-{c.name}'

        p_tex, p_err = render_pyomo_math(c, spec, constr_emap)
        g_tex, g_err = derive_gurobi_math(c, spec, constr_emap)

        raw_pyomo = c.raw_source.strip()
        pyomo_html = _tag_code(raw_pyomo, constr_emap)

        raw_gurobi = slices.get(c.name, '')
        gurobi_panel_html = _render_gurobi_panel(raw_gurobi, gurobi_emap)

        # Walkthrough
        walkthrough_html: Optional[str] = None
        if ns:
            try:
                wt = build_walkthrough(c, spec, generated_code, ns)
                walkthrough_html = _render_walkthrough(wt)
            except Exception as e:
                walkthrough_html = (
                    f'<div class="walkthrough"><div class="step-err">'
                    f'⚠ Walkthrough builder failed for {_escape(c.name)}: '
                    f'{_escape(type(e).__name__ + ": " + str(e))}'
                    f'</div></div>')
        elif capture_err:
            walkthrough_html = (
                f'<div class="walkthrough"><div class="step-err">'
                f'⚠ Could not run synthetic data through generated code — '
                f'{_escape(capture_err)}</div></div>')

        cards.append(Card(
            name=c.name,
            kind='constraint',
            pyomo_code_html=pyomo_html,
            pyomo_math_tex=p_tex,
            pyomo_math_err=p_err,
            gurobi_math_tex=g_tex,
            gurobi_math_err=g_err,
            gurobi_code_html=gurobi_panel_html,
            walkthrough_html=walkthrough_html,
            name_mapping_html=name_mapping_html,
        ))

    if spec.obj is not None:
        o = spec.obj
        obj_emap = dict(pyomo_emap)
        obj_emap[o.name] = f'obj-{o.name}'
        gurobi_emap = _build_gurobi_emap(spec, '__objective__')

        p_tex, p_err = render_obj_math(o, obj_emap)
        g_tex, g_err = derive_gurobi_obj_math(o, spec, obj_emap)

        raw_pyomo = o.raw_source.strip()
        pyomo_html = _tag_code(raw_pyomo, obj_emap)

        raw_gurobi = slices.get('__objective__', '')
        gurobi_panel_html = _render_gurobi_panel(raw_gurobi, gurobi_emap)

        cards.append(Card(
            name=o.name,
            kind='objective',
            pyomo_code_html=pyomo_html,
            pyomo_math_tex=p_tex,
            pyomo_math_err=p_err,
            gurobi_math_tex=g_tex,
            gurobi_math_err=g_err,
            gurobi_code_html=gurobi_panel_html,
            walkthrough_html=None,
            name_mapping_html=name_mapping_html,
        ))

    return cards


# ===================================================================
# 8. HTML rendering
# ===================================================================

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pyomo→Gurobi pipeline</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer
  src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer
  src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
      ],
      trust: true,
      strict: false
    });
    initHighlighting();
  "></script>
<style>
:root {
  color-scheme: light;
  --c-set:    #0550ae;
  --c-var:    #6e40c9;
  --c-param:  #953800;
  --c-constr: #1a7f37;
  --c-border: #d0d7de;
  --c-bg:     #f6f8fa;
  --c-panel:  #ffffff;
  --c-text:   #1f2328;
  --c-muted:  #57606a;
}
html, body {
  background: var(--c-bg);
  color: var(--c-text);
}
body {
  font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  max-width: 1200px;
  margin: 2rem auto;
  padding: 0 1rem;
  line-height: 1.5;
}
h1 { font-size: 1.5rem; margin-bottom: 0.2rem; }
.subtitle { color: var(--c-muted); margin-bottom: 1.5rem; font-size: 0.95rem; }

/* Legend */
.legend {
  display: flex; gap: 1.2rem; align-items: center;
  background: var(--c-panel); border: 1px solid var(--c-border);
  border-radius: 8px; padding: 0.5rem 1rem;
  margin-bottom: 1.5rem; flex-wrap: wrap;
}
.legend-item { display: flex; align-items: center; gap: 0.35rem; font-size: 0.85rem; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.ld-set    { background: var(--c-set); }
.ld-var    { background: var(--c-var); }
.ld-param  { background: var(--c-param); }
.ld-constr { background: var(--c-constr); }

/* Card */
.card {
  border: 1px solid var(--c-border);
  border-radius: 10px;
  margin-bottom: 1.5rem;
  background: var(--c-panel);
  overflow: hidden;
  transition: border-color .15s;
}
.card.approved { border-color: #2da44e; }
.card.rejected  { border-color: #cf222e; }

.card-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.6rem 1rem;
  background: var(--c-bg);
  border-bottom: 1px solid var(--c-border);
}
.card-head h2 { font-size: 1.05rem; margin: 0; }
.kind-tag {
  font-size: 0.75rem; padding: 2px 7px; border-radius: 4px; font-weight: 600;
}
.kind-tag.objective  { background: #fff3cf; color: #5a4500; }
.kind-tag.constraint { background: #ddf4ff; color: #0a4f80; }

/* Layout (b): four boxes in 2×2 grid up top, walkthrough full-width below */
.panels-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
}
.panels-grid .panel-pyomo-code  { border-right: 1px solid var(--c-border); border-bottom: 1px solid var(--c-border); }
.panels-grid .panel-pyomo-math  { border-bottom: 1px solid var(--c-border); }
.panels-grid .panel-gurobi-math { border-right: 1px solid var(--c-border); }

.panel {
  padding: 0.8rem 1rem;
  min-height: 80px;
}
.panel-label {
  font-size: 0.7rem; font-weight: 700; color: var(--c-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 0.45rem;
}
.panel pre {
  margin: 0; font-size: 0.8rem; line-height: 1.45;
  white-space: pre-wrap; word-break: break-word;
  background: none; padding: 0; color: var(--c-text);
  font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
}
.math-panel {
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; text-align: center;
}
.math-panel .katex { color: var(--c-text) !important; }
.math-err {
  background: #fff8c5; border: 1px solid #d4a72c;
  border-radius: 4px; padding: 0.3rem 0.6rem;
  font-size: 0.83rem; color: #5a4500;
}

/* Gurobi code micro-sections */
.gsec-header {
  font-size: 0.72rem; font-weight: 700;
  color: var(--c-muted);
  margin: 0.6rem 0 0.15rem;
  padding: 0.12rem 0.5rem;
  background: var(--c-bg);
  border-left: 3px solid var(--c-border);
  border-radius: 0 3px 3px 0;
  display: block;
}
.gsec-header:first-child { margin-top: 0; }
.gsec-pre {
  margin: 0 0 0.1rem; font-size: 0.8rem; line-height: 1.45;
  white-space: pre-wrap; word-break: break-word;
  background: none; padding: 0 0 0 0.5rem; color: var(--c-text);
  font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
  border-left: 3px solid transparent;
}

/* Name mapping band (only when names diverge) */
.name-mapping {
  padding: 0.45rem 1rem;
  background: #fff9e6;
  border-top: 1px solid var(--c-border);
  border-bottom: 1px solid var(--c-border);
  font-size: 0.78rem;
}
.nm-label { font-weight: 700; color: var(--c-muted); margin-right: 0.5rem; }
.nm-row { display: inline-block; margin-right: 0.6rem; white-space: nowrap; }
.nm-arrow { color: var(--c-muted); margin: 0 0.2rem; }
.nm-data, .nm-code {
  font-family: ui-monospace, monospace; font-size: 0.85em;
  color: var(--c-text); background: #f3f3f3;
  padding: 1px 5px; border-radius: 3px;
}

/* Walkthrough section */
.walkthrough {
  padding: 1rem 1.2rem 1.2rem;
  background: linear-gradient(180deg, #fafbfc 0%, #ffffff 60px);
  border-top: 1px solid var(--c-border);
}
.walkthrough-title {
  font-size: 0.72rem; font-weight: 700;
  color: var(--c-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 0.7rem;
}
.step {
  margin-bottom: 1.1rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px dashed var(--c-border);
}
.step:last-child { border-bottom: none; padding-bottom: 0; margin-bottom: 0; }
.step.step-failed .step-body { opacity: 0.85; }
.step-head { margin-bottom: 0.5rem; line-height: 1.5; }
.step-num {
  display: inline-block; font-size: 0.72rem; font-weight: 700;
  color: #fff; background: var(--c-muted);
  padding: 1px 7px; border-radius: 3px; margin-right: 0.4rem;
  letter-spacing: 0.04em;
}
.step.step-failed .step-num { background: #cf6e22; }
.step-title {
  font-weight: 700; font-size: 0.95rem; color: var(--c-text);
  margin-right: 0.4rem;
}
.step-sub { color: var(--c-muted); font-size: 0.85rem; }
.step-body { padding-left: 0.4rem; }
.step-label {
  font-size: 0.82rem; color: var(--c-muted);
  margin-bottom: 0.3rem;
}
.step-err {
  background: #fff8c5; border: 1px solid #d4a72c;
  border-radius: 4px; padding: 0.4rem 0.7rem;
  font-size: 0.85rem; color: #5a4500;
}
.step-block-err {
  background: #fff8c5; border: 1px solid #d4a72c;
  border-radius: 4px; padding: 0.3rem 0.6rem;
  font-size: 0.8rem; color: #5a4500; margin-bottom: 0.4rem;
}
.step1-body, .step3-body {
  display: flex; flex-wrap: wrap; gap: 1.5rem 2rem;
  align-items: flex-start;
}
.var-table-wrap, .merge-table-wrap { flex: 0 0 auto; }
.subscript {
  font-style: italic; color: var(--c-muted); font-weight: 400;
  font-size: 0.85em; margin-left: 1px;
}
.ent-token { padding: 0 1px; }

/* Walkthrough tables (Steps 1, 2, 3) */
.wt-table {
  border-collapse: collapse;
  font-size: 0.82rem;
  font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
}
.wt-table th, .wt-table td {
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--c-border);
  white-space: nowrap;
}
.wt-table thead th {
  background: var(--c-bg); font-weight: 700;
  text-align: left;
}
.wt-th-axis  { color: var(--c-set); }
.wt-th-int   { color: var(--c-muted); text-align: right; }
.wt-th-val   { color: var(--c-param); text-align: right; }
.wt-td-axis  { color: var(--c-set); }
.wt-td-int   { color: var(--c-muted); text-align: right; font-weight: 600; }
.wt-td-val   { color: var(--c-param); text-align: right; font-weight: 600; }

/* Step 4: matrix */
.step4-body { overflow-x: auto; }
.wt-matrix {
  border-collapse: collapse;
  font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
  font-size: 0.82rem;
}
.wt-matrix th, .wt-matrix td {
  border: 1px solid var(--c-border);
  padding: 0.2rem 0.45rem;
  text-align: center;
  min-width: 36px;
}
.wt-matrix .mtx-corner {
  background: var(--c-bg);
  border: 1px solid var(--c-border);
}
.wt-matrix .mtx-var-head {
  background: #ece7f7;
  font-weight: 700;
  text-align: center;
  border-bottom: 2px solid var(--c-var);
}
.wt-matrix .mtx-col-sub {
  background: #f7f5fc;
  font-weight: 600;
  font-size: 0.78rem;
  padding: 0.2rem 0.4rem;
}
.mtx-sub { display: block; font-style: italic; }
.mtx-colnum { display: block; color: var(--c-muted); font-size: 0.7rem; font-weight: 400; }
.wt-matrix .mtx-row-sub {
  background: #e7f3ea;
  font-weight: 700;
  text-align: right;
  padding-right: 0.55rem;
  border-right: 2px solid var(--c-constr);
}
.wt-matrix .mtx-rownum {
  background: #f3f8f4;
  color: var(--c-muted);
  font-size: 0.78rem;
  font-weight: 400;
  text-align: right;
  border-right: 1px solid var(--c-border);
}
.wt-matrix .mtx-zero {
  color: #d0d7de;
  background: #fdfdfd;
}
.wt-matrix .mtx-cell { font-weight: 700; }
.wt-matrix .mtx-pos { background: #fff5e6; color: #953800; }
.wt-matrix .mtx-neg { background: #fde8e8; color: #cf222e; }

/* Entity colors — applies to spans, table cells, all elements */
[class*="ent-set-"]    { color: var(--c-set);    cursor: pointer; font-weight: 600; }
[class*="ent-var-"]    { color: var(--c-var);    cursor: pointer; font-weight: 600; }
[class*="ent-param-"]  { color: var(--c-param);  cursor: pointer; font-weight: 600; }
[class*="ent-constr-"] { color: var(--c-constr); cursor: pointer; font-weight: 600; }
[class*="ent-obj-"]    { color: var(--c-constr); cursor: pointer; font-weight: 600; }
/* Override muted/colored cells when they carry an entity class for hover */
td[class*="ent-var-"]    { color: var(--c-var); }
td[class*="ent-constr-"] { color: var(--c-constr); }
td[class*="ent-param-"]  { color: var(--c-param); }
td[class*="ent-set-"]    { color: var(--c-set); }
th[class*="ent-var-"]    { color: var(--c-var); }
th[class*="ent-constr-"] { color: var(--c-constr); }

/* Highlighting */
.hl-hover  { background: rgba(255, 200, 50, 0.28) !important; border-radius: 2px; }
.hl-sticky { background: rgba(255, 140, 20, 0.38) !important; border-radius: 2px;
             outline: 1px dashed rgba(200, 100, 0, 0.5); }

/* Approval row */
.approval-row {
  display: flex; gap: 0.5rem; align-items: center;
  padding: 0.7rem 1rem;
  border-top: 1px dashed var(--c-border);
}
.btn { font-size: 0.9rem; padding: 0.35rem 0.9rem; border-radius: 6px; border: 1px solid;
       cursor: pointer; font-weight: 600; }
.btn-good { background: #2da44e; color: #fff; border-color: #2da44e; }
.btn-good:hover { background: #2c974b; }
.btn-bad  { background: #fff; color: #cf222e; border-color: #cf222e; }
.btn-bad:hover  { background: #fff5f5; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.feedback-box { display: none; padding: 0 1rem 0.7rem; }
.feedback-box textarea {
  width: 100%; min-height: 55px; box-sizing: border-box; font: inherit;
  padding: 0.35rem; border: 1px solid #999; border-radius: 4px;
  background: var(--c-panel); color: var(--c-text);
}

/* Progress bar */
.progress {
  position: sticky; bottom: 1rem;
  background: #1f2328; color: #fff;
  padding: 0.45rem 1rem; border-radius: 8px;
  font-size: 0.88rem; display: inline-block;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.status-msg { margin-top: 1rem; font-weight: 600; }
.status-msg.ok   { color: #2da44e; }
.status-msg.fail { color: #cf222e; }
</style>
</head>
<body>
<h1>Pyomo → Gurobi pipeline</h1>
<div class="subtitle">{{SUBTITLE}}</div>

<div class="legend">
  <strong style="font-size:0.85rem">Entities:</strong>
  <span class="legend-item"><span class="legend-dot ld-set"></span>set</span>
  <span class="legend-item"><span class="legend-dot ld-var"></span>variable</span>
  <span class="legend-item"><span class="legend-dot ld-param"></span>parameter</span>
  <span class="legend-item"><span class="legend-dot ld-constr"></span>constraint</span>
  <span style="color:var(--c-muted);font-size:0.82rem;margin-left:0.5rem">
    hover = highlight across panels · click = pin</span>
</div>

{{CARDS}}

<div id="progress-bar" class="progress" style="display:{{PROGRESS_DISPLAY}}">
  <span id="progress-text">0 / {{N_CARDS}} approved</span>
</div>
<div id="status-msg" class="status-msg"></div>

<script>
const NEEDS_APPROVAL = {{NEEDS_APPROVAL_JS}};
const N_CARDS = {{N_CARDS}};
const decisions = {};

function initHighlighting() {
  let sticky = null;

  function getEnt(el) {
    for (const c of el.classList) {
      if (c.startsWith('ent-')) return c;
    }
    return null;
  }

  function highlight(entCls) {
    document.querySelectorAll('.' + entCls).forEach(x => x.classList.add('hl-hover'));
  }
  function unhighlight() {
    document.querySelectorAll('.hl-hover').forEach(x => x.classList.remove('hl-hover'));
  }
  function pinSticky(entCls) {
    document.querySelectorAll('.hl-sticky').forEach(x => x.classList.remove('hl-sticky'));
    if (sticky === entCls) { sticky = null; return; }
    document.querySelectorAll('.' + entCls).forEach(x => x.classList.add('hl-sticky'));
    sticky = entCls;
  }

  // Attach to all entity spans (including KaTeX-rendered ones)
  document.querySelectorAll('[class]').forEach(el => {
    const ent = getEnt(el);
    if (!ent) return;
    el.addEventListener('mouseenter', e => { e.stopPropagation(); highlight(ent); });
    el.addEventListener('mouseleave', e => { e.stopPropagation(); unhighlight(); });
    el.addEventListener('click', e => { e.stopPropagation(); pinSticky(ent); });
  });

  document.addEventListener('click', () => {
    if (sticky) {
      document.querySelectorAll('.hl-sticky').forEach(x => x.classList.remove('hl-sticky'));
      sticky = null;
    }
  });
}

function update_progress() {
  const n = Object.values(decisions).filter(d => d === 'ok').length;
  const el = document.getElementById('progress-text');
  if (el) el.textContent = n + ' / ' + N_CARDS + ' approved';
  if (n === N_CARDS) {
    fetch('/done', {method: 'POST'}).then(() => {
      const m = document.getElementById('status-msg');
      m.textContent = 'All approved. You can close this tab.';
      m.className = 'status-msg ok';
    });
  }
}

function on_approve(name) {
  decisions[name] = 'ok';
  const card = document.getElementById('card-' + name);
  card.classList.add('approved'); card.classList.remove('rejected');
  card.querySelectorAll('button').forEach(b => b.disabled = true);
  if (NEEDS_APPROVAL) fetch('/approve', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
  update_progress();
}

function on_reject(name) {
  const fb = document.getElementById('feedback-' + name);
  fb.style.display = 'block';
  fb.querySelector('textarea').focus();
}

function on_submit_reject(name) {
  const text = document.getElementById('feedback-' + name).querySelector('textarea').value;
  decisions[name] = 'reject';
  document.getElementById('card-' + name).classList.add('rejected');
  document.getElementById('card-' + name).querySelectorAll('button').forEach(b => b.disabled = true);
  fetch('/reject', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, description: text})})
    .then(r => r.json()).then(j => {
      const m = document.getElementById('status-msg');
      m.textContent = 'Bug report saved to ' + j.report_path + '. Returning error.';
      m.className = 'status-msg fail';
    });
}
</script>
</body>
</html>
"""


def _math_block(tex: str, err: Optional[str]) -> str:
    if err or not tex:
        msg = _escape(err or 'no expression')
        return f'<div class="math-err">⚠ {msg}</div>'
    return f'<div>$$ {tex} $$</div>'


def _render_card(card: Card, needs_approval: bool) -> str:
    kind_cls = 'objective' if card.kind == 'objective' else 'constraint'

    approval = ''
    if needs_approval:
        safe = card.name.replace("'", "\\'")
        approval = f"""
<div class="approval-row">
  <button class="btn btn-good" onclick="on_approve('{safe}')">Looks correct</button>
  <button class="btn btn-bad"  onclick="on_reject('{safe}')">Something's wrong</button>
</div>
<div class="feedback-box" id="feedback-{card.name}">
  <textarea placeholder="What's wrong? (optional)"></textarea>
  <div style="margin-top:0.4rem">
    <button class="btn btn-bad" onclick="on_submit_reject('{safe}')">Submit &amp; halt</button>
  </div>
</div>"""

    panels = f"""
    <div class="panel panel-pyomo-code">
      <div class="panel-label">Pyomo code</div>
      <pre>{card.pyomo_code_html}</pre>
    </div>
    <div class="panel panel-pyomo-math math-panel">
      <div class="panel-label" style="align-self:flex-start">Pyomo math</div>
      {_math_block(card.pyomo_math_tex, card.pyomo_math_err)}
    </div>
    <div class="panel panel-gurobi-math math-panel">
      <div class="panel-label" style="align-self:flex-start">Gurobi math</div>
      {_math_block(card.gurobi_math_tex, card.gurobi_math_err)}
    </div>
    <div class="panel panel-gurobi-code">
      <div class="panel-label">Gurobi code</div>
      {card.gurobi_code_html}
    </div>"""

    name_mapping = card.name_mapping_html or ''

    walkthrough_section = ''
    if card.walkthrough_html is not None:
        walkthrough_section = card.walkthrough_html

    return f"""
<div class="card" id="card-{card.name}">
  <div class="card-head">
    <h2>{_escape(card.name)}</h2>
    <span class="kind-tag {kind_cls}">{card.kind}</span>
  </div>
  <div class="panels-grid">
    {panels}
  </div>
  {name_mapping}
  {walkthrough_section}
  {approval}
</div>"""


def _render_html(cards: list[Card], needs_approval: bool, subtitle: str) -> str:
    cards_html = '\n'.join(_render_card(c, needs_approval) for c in cards)
    return (_HTML_TEMPLATE
            .replace('{{SUBTITLE}}', subtitle)
            .replace('{{CARDS}}', cards_html)
            .replace('{{N_CARDS}}', str(len(cards)))
            .replace('{{NEEDS_APPROVAL_JS}}', 'true' if needs_approval else 'false')
            .replace('{{PROGRESS_DISPLAY}}', 'inline-block' if needs_approval else 'none')
            )


# ===================================================================
# 9. Approval HTTP server
# ===================================================================

class _ApprovalState:
    def __init__(self):
        self.approved: set[str] = set()
        self.rejection: Optional[dict] = None
        self.done = threading.Event()


def _serve_approval(html: str, cards: list[Card], output_dir: str,
                    pyomo_source: str, generated_code: str) -> _ApprovalState:
    state = _ApprovalState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                body = html.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b''
            try:
                payload = json.loads(raw.decode('utf-8')) if raw else {}
            except Exception:
                payload = {}
            if self.path == '/approve':
                state.approved.add(payload.get('name', ''))
                self._json({'ok': True})
                if len(state.approved) >= len(cards):
                    state.done.set()
            elif self.path == '/reject':
                rp = _save_bug_report(output_dir, payload.get('name', ''),
                                      payload.get('description', ''),
                                      pyomo_source, generated_code, cards)
                state.rejection = {'name': payload.get('name', ''),
                                   'description': payload.get('description', ''),
                                   'report_path': rp}
                self._json({'ok': True, 'report_path': rp})
                state.done.set()
            elif self.path == '/done':
                self._json({'ok': True})
                state.done.set()
            else:
                self.send_response(404); self.end_headers()

        def _json(self, obj):
            body = json.dumps(obj).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f'http://127.0.0.1:{port}/'
    print(f'[visualize] approval UI at {url}')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        state.done.wait()
    except KeyboardInterrupt:
        pass
    time.sleep(0.5)
    server.shutdown()
    return state


def _save_bug_report(output_dir: str, name: str, description: str,
                     pyomo_source: str, generated_code: str,
                     cards: list[Card]) -> str:
    bugs = os.path.join(output_dir, 'bugs')
    os.makedirs(bugs, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe = ''.join(c if c.isalnum() else '_' for c in name) or 'unknown'
    path = os.path.join(bugs, f'{ts}_{safe}.json')
    fc = next((c for c in cards if c.name == name), None)
    payload = {
        'timestamp': datetime.now().isoformat(),
        'failed_constraint': name,
        'user_description': description,
        'pyomo_source': pyomo_source,
        'generated_gurobi_code': generated_code,
        'failed_card': {
            'kind': fc.kind,
            'pyomo_math': fc.pyomo_math_tex,
            'gurobi_math': fc.gurobi_math_tex,
        } if fc else None,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


# ===================================================================
# 10. Public entry point
# ===================================================================

class TranslationRejected(Exception):
    def __init__(self, info: dict):
        self.info = info
        super().__init__(f'Translation rejected on {info.get("name")!r}: '
                         f'report at {info.get("report_path")}')


def visualize(build_pyomo_fn, needs_approval: bool = True,
              output_dir: str = 'visualize_output',
              auto_open: bool = True) -> dict:
    """
    Render the four-panel pipeline report and optionally serve for approval.

    Args:
      build_pyomo_fn: function `build_pyomo_model(data)` from an example module.
      needs_approval: if True, open browser UI and block until approved/rejected.
      output_dir: directory for HTML report and bug reports.
      auto_open: open report in browser in static mode.
    """
    from translator import translate

    os.makedirs(output_dir, exist_ok=True)
    spec = extract_spec(build_pyomo_fn)
    generated_code = translate(build_pyomo_fn)

    cards = _build_cards(spec, generated_code, build_pyomo_fn)

    subtitle = ('Review each card — hover/click to highlight matching entities.'
                if not needs_approval else
                'Review each card, then approve or flag. Hover/click to highlight entities.')
    html = _render_html(cards, needs_approval, subtitle)

    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    if not needs_approval:
        if auto_open:
            try:
                webbrowser.open(f'file://{os.path.abspath(report_path)}')
            except Exception:
                pass
        return {'status': 'static_report', 'report_path': report_path,
                'n_cards': len(cards)}

    state = _serve_approval(html, cards, output_dir, spec.func_source, generated_code)
    if state.rejection is not None:
        raise TranslationRejected(state.rejection)
    return {'status': 'approved', 'report_path': report_path, 'n_cards': len(cards)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python visualize.py <example_module> [--no-approval]')
        sys.exit(2)
    import importlib
    mod = importlib.import_module(sys.argv[1])
    needs = '--no-approval' not in sys.argv[2:]
    visualize(mod.build_pyomo_model, needs_approval=needs)
