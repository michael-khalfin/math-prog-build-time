"""
Pyomo → gurobipy-pandas translator.

Public API:
    from translator import translate
    code_str = translate(build_pyomo_model)
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SetInfo:
    pyomo_name: str
    data_key: str
    dimen: int = 1
    is_indexed: bool = False       # pyo.Set(m.X, initialize=...)
    index_set: Optional[str] = None
    is_subset: bool = False        # declared with within=
    within_set: Optional[str] = None


@dataclass
class VarInfo:
    pyomo_name: str
    index_sets: list                # ordered Pyomo set names
    vtype: str = 'CONTINUOUS'       # 'CONTINUOUS' | 'INTEGER'


@dataclass
class ParamInfo:
    pyomo_name: str
    index_sets: list                # empty = scalar
    data_key: str


@dataclass
class SumTermInfo:
    var_name: str
    param_name: Optional[str]
    loop_var: object               # str or list[str] for tuple destructuring
    iter_set: str
    iter_is_indexed: bool
    iter_index_arg: Optional[str]  # subscript arg when indexed
    var_subscript_args: list = field(default_factory=list)  # args in m.var[a, b, c]


@dataclass
class ConstrInfo:
    pyomo_name: str
    index_sets: list
    rule_name: str
    rule_args: list                # args after 'm'
    pattern: str = ''              # P1–P6, set by classifier
    lhs_terms: list = field(default_factory=list)
    rhs_node: object = None        # AST node for RHS
    op: str = ''                   # 'LEQ' | 'GEQ' | 'EQ'
    flow_sub: bool = False         # True for P3 (lhs1 - lhs2)
    lhs_direct_var: Optional[str] = None
    # For P3: the two intermediate variable names used in flow_out - flow_in
    flow_term_names: list = field(default_factory=list)
    # For P4 where LHS is a direct var (component balance) — swap sides
    lhs_is_direct_var: bool = False
    rhs_terms: list = field(default_factory=list)  # RHS sum terms when swapped
    # For pyo.Constraint(expr=...) — direct expression, no rule function
    inline_expr: object = None


@dataclass
class ObjInfo:
    pyomo_name: str
    sense: str                     # 'MINIMIZE' | 'MAXIMIZE'
    lhs_terms: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(node, attr, default=None):
    return getattr(node, attr, default)


def _node_is_m_attr(node) -> Optional[str]:
    """If node is `m.Something`, return 'Something', else None."""
    if (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'm'):
        return node.attr
    return None


def _node_is_data_subscript(node) -> Optional[str]:
    """If node is `data['key']` or `data["key"]`, return 'key', else None."""
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'data'):
        sl = node.slice
        # Python 3.9+ ast.Subscript.slice is direct; older wraps in ast.Index
        if isinstance(sl, ast.Index):
            sl = sl.value
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return None


def _extract_keyword(call_node, kw_name):
    for kw in call_node.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _extract_subscript_args(subscript_node: ast.Subscript) -> list:
    """Return identifier names from m.var[a, b, c] — used to map loop vars to index positions."""
    sl = subscript_node.slice
    if isinstance(sl, ast.Index):   # Python < 3.9 wraps in ast.Index
        sl = sl.value
    if isinstance(sl, ast.Tuple):
        return [e.id for e in sl.elts if isinstance(e, ast.Name)]
    if isinstance(sl, ast.Name):
        return [sl.id]
    return []


def _op_str(op_node) -> str:
    if isinstance(op_node, ast.LtE):
        return 'LEQ'
    if isinstance(op_node, ast.GtE):
        return 'GEQ'
    if isinstance(op_node, ast.Eq):
        return 'EQ'
    raise NotImplementedError(f"Unsupported comparison operator: {type(op_node).__name__}")


def _gurobi_sense(op: str) -> str:
    return {'LEQ': 'gp.GRB.LESS_EQUAL', 'GEQ': 'gp.GRB.GREATER_EQUAL', 'EQ': 'gp.GRB.EQUAL'}[op]


def _py_op_str(op: str) -> str:
    return {'LEQ': '<=', 'GEQ': '>=', 'EQ': '=='}[op]


# ---------------------------------------------------------------------------
# Stage 2: Parser
# ---------------------------------------------------------------------------

class _Translator:
    def __init__(self, func_def: ast.FunctionDef):
        self.func_def = func_def
        self.sets: dict[str, SetInfo] = {}
        self.vars: dict[str, VarInfo] = {}
        self.params: dict[str, ParamInfo] = {}
        self.constrs: list[ConstrInfo] = []
        self.obj: Optional[ObjInfo] = None
        self._rules: dict[str, ast.FunctionDef] = {}
        # Declaration order for vars/params (for codegen ordering)
        self._var_order: list[str] = []
        self._param_order: list[str] = []

    # ------------------------------------------------------------------
    def parse(self):
        for stmt in self.func_def.body:
            if isinstance(stmt, ast.FunctionDef):
                self._rules[stmt.name] = stmt
            elif isinstance(stmt, ast.Assign):
                self._parse_assignment(stmt)

    def _parse_assignment(self, stmt: ast.Assign):
        # Target must be m.X
        if len(stmt.targets) != 1:
            return
        tgt = stmt.targets[0]
        if not isinstance(tgt, ast.Attribute):
            return
        if not (isinstance(tgt.value, ast.Name) and tgt.value.id == 'm'):
            return
        attr_name = tgt.attr

        val = stmt.value
        if not isinstance(val, ast.Call):
            return
        func = val.func
        # Must be pyo.Something
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == 'pyo'):
            return
        kind = func.attr

        if kind == 'Set':
            self._parse_set(attr_name, val)
        elif kind == 'Param':
            self._parse_param(attr_name, val)
        elif kind == 'Var':
            self._parse_var(attr_name, val)
        elif kind == 'Constraint':
            self._parse_constraint(attr_name, val)
        elif kind == 'Objective':
            self._parse_objective(attr_name, val)

    def _parse_set(self, name: str, call: ast.Call):
        data_key = None
        init_kw = _extract_keyword(call, 'initialize')
        if init_kw is not None:
            data_key = _node_is_data_subscript(init_kw) or name

        dimen = 1
        dimen_kw = _extract_keyword(call, 'dimen')
        if dimen_kw is not None and isinstance(dimen_kw, ast.Constant):
            dimen = dimen_kw.value

        is_indexed = False
        index_set = None
        # Positional args: first arg being m.X means indexed set
        if call.args:
            first_arg = call.args[0]
            m_attr = _node_is_m_attr(first_arg)
            if m_attr:
                is_indexed = True
                index_set = m_attr

        is_subset = False
        within_set = None
        within_kw = _extract_keyword(call, 'within')
        if within_kw is not None:
            m_attr = _node_is_m_attr(within_kw)
            if m_attr:
                is_subset = True
                within_set = m_attr

        self.sets[name] = SetInfo(
            pyomo_name=name,
            data_key=data_key or name,
            dimen=dimen,
            is_indexed=is_indexed,
            index_set=index_set,
            is_subset=is_subset,
            within_set=within_set,
        )

    def _parse_param(self, name: str, call: ast.Call):
        # Positional args are index sets
        index_sets = []
        for arg in call.args:
            m_attr = _node_is_m_attr(arg)
            if m_attr:
                index_sets.append(m_attr)

        init_kw = _extract_keyword(call, 'initialize')
        data_key = None
        if init_kw is not None:
            data_key = _node_is_data_subscript(init_kw)
        if data_key is None:
            data_key = name

        self.params[name] = ParamInfo(pyomo_name=name, index_sets=index_sets, data_key=data_key)
        self._param_order.append(name)

    def _parse_var(self, name: str, call: ast.Call):
        index_sets = []
        for arg in call.args:
            m_attr = _node_is_m_attr(arg)
            if m_attr:
                index_sets.append(m_attr)

        vtype = 'CONTINUOUS'
        domain_kw = _extract_keyword(call, 'domain')
        if domain_kw is not None:
            d = _node_is_m_attr(domain_kw)
            if d is None and isinstance(domain_kw, ast.Attribute):
                d = domain_kw.attr
            if d:
                if 'Integer' in d:
                    vtype = 'INTEGER'
                elif 'Binary' in d:
                    vtype = 'BINARY'

        self.vars[name] = VarInfo(pyomo_name=name, index_sets=index_sets, vtype=vtype)
        self._var_order.append(name)

    def _parse_constraint(self, name: str, call: ast.Call):
        index_sets = []
        for arg in call.args:
            m_attr = _node_is_m_attr(arg)
            if m_attr:
                index_sets.append(m_attr)

        rule_kw = _extract_keyword(call, 'rule')
        rule_name = rule_kw.id if isinstance(rule_kw, ast.Name) else ''

        # Get rule args (after 'm')
        rule_args = []
        if rule_name and rule_name in self._rules:
            rfunc = self._rules[rule_name]
            args = [a.arg for a in rfunc.args.args]
            rule_args = [a for a in args if a != 'm']

        # Check for expr= (scalar inline expression, no rule function)
        inline_expr = _extract_keyword(call, 'expr')

        self.constrs.append(ConstrInfo(
            pyomo_name=name,
            index_sets=index_sets,
            rule_name=rule_name,
            rule_args=rule_args,
            inline_expr=inline_expr,
        ))

    def _parse_objective(self, name: str, call: ast.Call):
        sense = 'MINIMIZE'
        sense_kw = _extract_keyword(call, 'sense')
        if sense_kw is not None:
            attr = _node_is_m_attr(sense_kw)
            if attr is None and isinstance(sense_kw, ast.Attribute):
                attr = sense_kw.attr
            if attr and 'maximize' in attr.lower():
                sense = 'MAXIMIZE'

        rule_kw = _extract_keyword(call, 'rule')
        rule_name = rule_kw.id if isinstance(rule_kw, ast.Name) else ''

        self.obj = ObjInfo(pyomo_name=name, sense=sense)
        self.obj._rule_name = rule_name

    # ------------------------------------------------------------------
    def classify(self):
        classifier = _RuleClassifier(self)
        for ci in self.constrs:
            if ci.inline_expr is not None:
                classifier.classify_inline(ci)
            elif ci.rule_name and ci.rule_name in self._rules:
                classifier.classify_constr(ci, self._rules[ci.rule_name])
        if self.obj and hasattr(self.obj, '_rule_name'):
            rn = self.obj._rule_name
            if rn and rn in self._rules:
                classifier.classify_obj(self.obj, self._rules[rn])

    # ------------------------------------------------------------------
    def generate(self) -> str:
        # Build index registry first
        registry = _IndexRegistry(self)
        registry.build()
        gen = _CodeGen(self, registry)
        return gen.generate()


# ---------------------------------------------------------------------------
# Stage 3: Classifier
# ---------------------------------------------------------------------------

class _RuleClassifier:
    def __init__(self, translator: _Translator):
        self.t = translator

    def classify_constr(self, ci: ConstrInfo, rfunc: ast.FunctionDef):
        # Collect intermediate assigns (for P3 flow balance)
        assigns: dict[str, ast.expr] = {}
        return_node = None
        for stmt in rfunc.body:
            if isinstance(stmt, ast.Return):
                return_node = stmt.value
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    assigns[stmt.targets[0].id] = stmt.value

        if return_node is None:
            return

        if not isinstance(return_node, ast.Compare):
            return

        if len(return_node.ops) > 1:
            raise NotImplementedError(
                "Chained comparisons are not supported; use single binary comparisons only."
            )

        op = _op_str(return_node.ops[0])
        ci.op = op
        lhs_node = return_node.left
        rhs_node = return_node.comparators[0]
        ci.rhs_node = rhs_node

        # Resolve intermediate name references (e.g., coverage = sum(...); return coverage >= ...)
        if isinstance(lhs_node, ast.Name) and lhs_node.id in assigns:
            lhs_node = assigns[lhs_node.id]
        if isinstance(rhs_node, ast.Name) and rhs_node.id in assigns:
            ci.rhs_node = assigns[rhs_node.id]

        # --- Determine pattern ---

        # P3: BinOp(Sub) on LHS — handles both named intermediates and inline sums.
        # Named:  flow_out = sum(...); flow_in = sum(...); return flow_out - flow_in == rhs
        # Inline: return (sum(...) - sum(...)) == rhs
        if isinstance(lhs_node, ast.BinOp) and isinstance(lhs_node.op, ast.Sub):
            left_node  = lhs_node.left
            right_node = lhs_node.right
            # Resolve named intermediate variables to their call expressions
            if isinstance(left_node, ast.Name) and left_node.id in assigns:
                left_node = assigns[left_node.id]
            if isinstance(right_node, ast.Name) and right_node.id in assigns:
                right_node = assigns[right_node.id]
            left_term  = self._parse_sum_call(left_node)
            right_term = self._parse_sum_call(right_node)
            if left_term is not None and right_term is not None:
                ci.pattern = 'P3'
                ci.flow_sub = True
                ci.lhs_terms = [left_term, right_term]
                return

        # P4b / P6: LHS is m.var[...] (direct var access)
        lhs_m_var = self._extract_direct_var(lhs_node)
        if lhs_m_var is not None:
            # Check if RHS is a sum() → P4 component balance (swapped)
            rhs_sum = self._parse_sum_call(rhs_node)
            if rhs_sum is not None:
                ci.pattern = 'P4'
                ci.lhs_is_direct_var = True
                ci.lhs_direct_var = lhs_m_var
                ci.rhs_terms = [rhs_sum]
                return
            # Otherwise P6 direct var
            ci.pattern = 'P6'
            ci.lhs_direct_var = lhs_m_var
            return

        # P1/P2/P4/P5: LHS is a sum() call
        lhs_sum = self._parse_sum_call(lhs_node)
        if lhs_sum is not None:
            ci.lhs_terms.append(lhs_sum)
            if lhs_sum.iter_is_indexed:
                if lhs_sum.param_name:
                    # Indexed set + param*var → cross-dim merge (P4)
                    ci.pattern = 'P4'
                else:
                    # Indexed set + pure var → rolling window / graph adjacency (P5)
                    ci.pattern = 'P5'
            elif not ci.rule_args:
                ci.pattern = 'P2'
            else:
                ci.pattern = 'P1'
            return

        # Fallback: treat as P6
        ci.pattern = 'P6'

    def classify_inline(self, ci: ConstrInfo):
        """Handle pyo.Constraint(expr=...) — a direct Compare expression."""
        expr_node = ci.inline_expr
        if not isinstance(expr_node, ast.Compare):
            return
        if len(expr_node.ops) > 1:
            raise NotImplementedError(
                "Chained comparisons are not supported; use single binary comparisons only."
            )
        ci.op = _op_str(expr_node.ops[0])
        lhs_node = expr_node.left
        ci.rhs_node = expr_node.comparators[0]

        lhs_sum = self._parse_sum_call(lhs_node)
        if lhs_sum is not None:
            ci.lhs_terms.append(lhs_sum)
            if lhs_sum.iter_is_indexed:
                ci.pattern = 'P4' if lhs_sum.param_name else 'P5'
            elif not ci.rule_args:
                ci.pattern = 'P2'
            else:
                ci.pattern = 'P1'
            return
        ci.pattern = 'P6'

    def classify_obj(self, obj: ObjInfo, rfunc: ast.FunctionDef):
        return_node = None
        for stmt in rfunc.body:
            if isinstance(stmt, ast.Return):
                return_node = stmt.value
        if return_node is None:
            return
        term = self._parse_sum_call(return_node)
        if term:
            obj.lhs_terms.append(term)
            # Handle nested generators: one SumTermInfo per generator
            # Re-parse to get all generators
            obj.lhs_terms = self._parse_all_generators(return_node)

    def _parse_all_generators(self, node: ast.expr) -> list:
        """Parse a sum() with potentially multiple for-clauses."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'sum'):
            return []
        if not node.args or not isinstance(node.args[0], ast.GeneratorExp):
            return []
        gen_exp = node.args[0]
        terms = []
        for comp in gen_exp.generators:
            term = self._parse_comprehension(gen_exp.elt, comp)
            if term:
                terms.append(term)
        return terms

    def _parse_sum_call(self, node: ast.expr) -> Optional[SumTermInfo]:
        """Parse a single sum(expr for x in Set) call."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'sum'):
            return None
        if not node.args or not isinstance(node.args[0], ast.GeneratorExp):
            return None
        gen_exp = node.args[0]
        if not gen_exp.generators:
            return None
        comp = gen_exp.generators[0]
        return self._parse_comprehension(gen_exp.elt, comp)

    def _parse_comprehension(self, elt: ast.expr, comp: ast.comprehension) -> Optional[SumTermInfo]:
        # Extract loop var
        if isinstance(comp.target, ast.Name):
            loop_var = comp.target.id
        elif isinstance(comp.target, ast.Tuple):
            loop_var = [e.id for e in comp.target.elts if isinstance(e, ast.Name)]
        else:
            return None

        # Extract iter set
        iter_node = comp.iter
        iter_is_indexed = False
        iter_index_arg = None
        iter_set = None

        m_attr = _node_is_m_attr(iter_node)
        if m_attr:
            iter_set = m_attr
        elif isinstance(iter_node, ast.Subscript):
            # m.Set[arg]
            m_attr = _node_is_m_attr(iter_node.value)
            if m_attr:
                iter_set = m_attr
                iter_is_indexed = True
                sl = iter_node.slice
                if isinstance(sl, ast.Index):
                    sl = sl.value
                if isinstance(sl, ast.Name):
                    iter_index_arg = sl.id
                elif isinstance(sl, ast.Tuple):
                    iter_index_arg = tuple(e.id for e in sl.elts if isinstance(e, ast.Name))
        if iter_set is None:
            return None

        # Extract var_name, param_name, and var_subscript_args from elt
        var_name = None
        param_name = None
        var_subscript_args = []

        if isinstance(elt, ast.Subscript):
            vn = _node_is_m_attr(elt.value)
            if vn and vn in self.t.vars:
                var_name = vn
                var_subscript_args = _extract_subscript_args(elt)
            elif vn and vn in self.t.params:
                param_name = vn
        elif isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.Mult):
            # param * var  or  var * param
            for side in [elt.left, elt.right]:
                if not isinstance(side, ast.Subscript):
                    continue
                attr = _node_is_m_attr(side.value)
                if attr is None:
                    continue
                if attr in self.t.vars:
                    var_name = attr
                    var_subscript_args = _extract_subscript_args(side)
                elif attr in self.t.params:
                    param_name = attr

        if var_name is None:
            return None

        return SumTermInfo(
            var_name=var_name,
            param_name=param_name,
            loop_var=loop_var,
            iter_set=iter_set,
            iter_is_indexed=iter_is_indexed,
            iter_index_arg=iter_index_arg,
            var_subscript_args=var_subscript_args,
        )

    def _extract_direct_var(self, node: ast.expr) -> Optional[str]:
        """If node is m.var[...] where var is a known Var, return var name."""
        if isinstance(node, ast.Subscript):
            m_attr = _node_is_m_attr(node.value)
            if m_attr and m_attr in self.t.vars:
                return m_attr
        return None


# ---------------------------------------------------------------------------
# Index Registry
# ---------------------------------------------------------------------------

class _IndexRegistry:
    """Maps Pyomo set name → list of index level names."""

    def __init__(self, translator: _Translator):
        self.t = translator
        self.registry: dict[str, list[str]] = {}

    def build(self):
        # Pass 1: scan constraint index_sets vs rule_args
        for ci in self.t.constrs:
            self._register_from_constr(ci)
        # Pass 2: scan sum term loop vars (registers iteration sets like OutArcs → ['j'])
        for ci in self.t.constrs:
            for term in ci.lhs_terms + ci.rhs_terms:
                self._register_loop_var(term)
        # Pass 3: for 2-D variable index sets that never appear as a constraint index
        # (e.g. Arcs in a P3 balance), derive dimension names from loop-var positions
        # recorded in var_subscript_args across all terms.
        self._register_var_dims_from_all_terms()

    def _register_from_constr(self, ci: ConstrInfo):
        rule_arg_cursor = 0
        for set_name in ci.index_sets:
            si = self.t.sets.get(set_name)
            dimen = si.dimen if si else 1
            names = []
            for _ in range(dimen):
                if rule_arg_cursor < len(ci.rule_args):
                    names.append(ci.rule_args[rule_arg_cursor])
                    rule_arg_cursor += 1
            if names and set_name not in self.registry:
                self.registry[set_name] = names

    def _register_loop_var(self, term: SumTermInfo):
        if term.iter_set not in self.registry:
            lv = term.loop_var
            if isinstance(lv, list):
                self.registry[term.iter_set] = lv
            else:
                self.registry[term.iter_set] = [lv]

    def _register_var_dims_from_all_terms(self):
        """Derive index-level names for variable index sets that never appear
        as constraint indices (e.g. a dimen=2 Arcs set used only in P3).
        For each term, positions in var_subscript_args that are NOT rule args
        are loop vars — they are the canonical dimension names for those positions."""
        # Accumulate: pos_names[var_name][abs_position] = canonical name
        pos_names: dict[str, dict[int, str]] = {}
        for ci in self.t.constrs:
            rule_args_set = set(ci.rule_args)
            for term in ci.lhs_terms + ci.rhs_terms:
                if not term.var_subscript_args:
                    continue
                vname = term.var_name
                if vname not in pos_names:
                    pos_names[vname] = {}
                for pos, arg in enumerate(term.var_subscript_args):
                    if arg not in rule_args_set and pos not in pos_names[vname]:
                        pos_names[vname][pos] = arg

        # Register set dimensions using the accumulated position names
        for vi in self.t.vars.values():
            if vi.pyomo_name not in pos_names:
                continue
            pmap = pos_names[vi.pyomo_name]
            cursor = 0
            for set_name in vi.index_sets:
                si = self.t.sets.get(set_name)
                dimen = si.dimen if si else 1
                if set_name not in self.registry:
                    names = [pmap.get(cursor + k) for k in range(dimen)]
                    if all(n is not None for n in names):
                        self.registry[set_name] = names
                cursor += dimen

    def names_for(self, set_name: str) -> list[str]:
        if set_name in self.registry:
            return self.registry[set_name]
        # Fallback: lowercase set name
        si = self.t.sets.get(set_name)
        if si and si.dimen == 1:
            return [set_name.lower()]
        if si and si.dimen == 2:
            return [set_name.lower() + '_0', set_name.lower() + '_1']
        return [set_name.lower()]

    def all_names_for_var(self, var: VarInfo) -> list[str]:
        names = []
        for s in var.index_sets:
            names.extend(self.names_for(s))
        return names


# ---------------------------------------------------------------------------
# Stage 4: Code Generator
# ---------------------------------------------------------------------------

class _CodeGen:
    def __init__(self, translator: _Translator, registry: _IndexRegistry):
        self.t = translator
        self.r = registry
        self.lines: list[str] = []

    def _emit(self, line: str = '', indent: int = 1):
        prefix = '    ' * indent
        self.lines.append(prefix + line if line else '')

    def generate(self) -> str:
        self.lines = []
        self._emit('def build_vectorized_model(data):', indent=0)
        self._emit('import gurobipy as gp')
        self._emit('import gurobipy_pandas as gppd')
        self._emit('import pandas as pd')
        self._emit('m = gp.Model()')
        self._emit()

        self._gen_vars()
        self._gen_params()

        if self.t.obj:
            self._gen_objective()

        for ci in self.t.constrs:
            self._gen_constraint(ci)

        self._emit('return m')
        return '\n'.join(self.lines)

    # ------------------------------------------------------------------
    def _gen_vars(self):
        for vname in self.t._var_order:
            vi = self.t.vars[vname]
            self._emit_var(vi)
            self._emit()

    def _emit_var(self, vi: VarInfo):
        all_names = self.r.all_names_for_var(vi)
        if vi.vtype == 'INTEGER':
            vtype_str = ', vtype=gp.GRB.INTEGER'
        elif vi.vtype == 'BINARY':
            vtype_str = ', vtype=gp.GRB.BINARY'
        else:
            vtype_str = ''

        # Determine if any set has dimen > 1
        has_tuple_set = any(
            (self.t.sets.get(s) or SetInfo('', '')).dimen > 1
            for s in vi.index_sets
        )

        df_var = f'df_{vi.pyomo_name}'
        idx_var = f'idx_{vi.pyomo_name}'

        names_repr = ', '.join(f"'{n}'" for n in all_names)

        if has_tuple_set:
            # Build list-of-tuples comprehension
            parts = []
            loop_parts = []
            for s in vi.index_sets:
                si = self.t.sets.get(s)
                dimen = si.dimen if si else 1
                if dimen > 1:
                    ns = self.r.names_for(s)
                    tup = ', '.join(ns)
                    loop_parts.append(f"({tup}) in data['{si.data_key}']")
                    parts.extend(ns)
                else:
                    n = self.r.names_for(s)[0]
                    loop_parts.append(f"{n} in data['{(si.data_key if si else s)}']")
                    parts.append(n)
            inner_tup = ', '.join(parts)
            loops = ' for '.join(loop_parts)
            self._emit(f"{idx_var}_tuples = [({inner_tup}) for {loops}]")
            self._emit(f"{idx_var} = pd.MultiIndex.from_tuples({idx_var}_tuples, names=[{names_repr}])")
        else:
            sets_repr_parts = []
            for s in vi.index_sets:
                si = self.t.sets.get(s)
                key = si.data_key if si else s
                sets_repr_parts.append(f"data['{key}']")
            sets_repr = ', '.join(sets_repr_parts)
            if len(vi.index_sets) == 1:
                self._emit(f"{idx_var} = pd.Index(data['{self.t.sets[vi.index_sets[0]].data_key}'], name='{all_names[0]}')")
            else:
                self._emit(f"{idx_var} = pd.MultiIndex.from_product([{sets_repr}], names=[{names_repr}])")

        self._emit(f"{df_var} = pd.DataFrame(index={idx_var})")
        self._emit(f"{df_var} = {df_var}.gppd.add_vars(m{vtype_str}, name='{vi.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_params(self):
        for pname in self.t._param_order:
            pi = self.t.params[pname]
            self._emit_param(pi)
        if self.t.params:
            self._emit()

    def _emit_param(self, pi: ParamInfo):
        var_name = f's_{pi.pyomo_name.lower()}'
        col_name = pi.pyomo_name.lower()
        if not pi.index_sets:
            # Scalar
            self._emit(f"{pi.pyomo_name.lower()} = data['{pi.data_key}']")
            return
        names = []
        for s in pi.index_sets:
            names.extend(self.r.names_for(s))
        if len(names) == 1:
            axis_repr = f"'{names[0]}'"
        else:
            axis_repr = '[' + ', '.join(f"'{n}'" for n in names) + ']'
        # name= is required so reset_index() and join() produce a named column
        self._emit(f"{var_name} = pd.Series(data['{pi.data_key}'], name='{col_name}').rename_axis({axis_repr})")

    def _param_var_name(self, pyomo_name: str) -> str:
        pi = self.t.params.get(pyomo_name)
        if pi and not pi.index_sets:
            return pyomo_name.lower()
        return f's_{pyomo_name.lower()}'

    # ------------------------------------------------------------------
    def _gen_objective(self):
        obj = self.t.obj
        if not obj.lhs_terms:
            return
        # Expect: sum(CompCost[c] * buy_comp[c, t] for c in C for t in T)
        # Pattern: param (1D over c) × var (2D over c, t)
        # Strategy: join param onto var df on the shared dimension, multiply, sum
        term = obj.lhs_terms[0]  # use first term for the main var
        vi = self.t.vars.get(term.var_name)
        pi = self.t.params.get(term.param_name) if term.param_name else None

        df_var = f'df_{term.var_name}'
        all_names = self.r.all_names_for_var(vi) if vi else []

        if pi and pi.index_sets:
            # join on shared dims
            shared_dims = self.r.names_for(pi.index_sets[0])
            on_dim = shared_dims[0]
            param_col = pi.pyomo_name.lower()
            self._emit(f"# Objective")
            self._emit(f"df_{term.var_name}_obj = {df_var}.join(s_{pi.pyomo_name.lower()}, on='{on_dim}')")
            self._emit(
                f"obj_expr = (df_{term.var_name}_obj['{param_col}'] * "
                f"df_{term.var_name}_obj['{term.var_name}']).sum()"
            )
        else:
            self._emit(f"# Objective")
            self._emit(f"obj_expr = {df_var}['{term.var_name}'].sum()")

        sense_str = f"gp.GRB.{'MINIMIZE' if obj.sense == 'MINIMIZE' else 'MAXIMIZE'}"
        self._emit(f"m.setObjective(obj_expr, {sense_str})")
        self._emit()

    # ------------------------------------------------------------------
    def _gen_constraint(self, ci: ConstrInfo):
        self._emit(f"# Constraint: {ci.pyomo_name}")
        if ci.pattern == 'P1':
            self._gen_P1(ci)
        elif ci.pattern == 'P2':
            self._gen_P2(ci)
        elif ci.pattern == 'P3':
            self._gen_P3(ci)
        elif ci.pattern == 'P4':
            self._gen_P4(ci)
        elif ci.pattern == 'P5':
            self._gen_P5(ci)
        elif ci.pattern == 'P6':
            self._gen_P6(ci)
        else:
            self._emit(f"# WARNING: unknown pattern for {ci.pyomo_name}")
        self._emit()

    # ------------------------------------------------------------------
    def _gen_P1(self, ci: ConstrInfo):
        term = ci.lhs_terms[0]
        vi = self.t.vars[term.var_name]
        df_var = f'df_{term.var_name}'
        lhs_var = f'lhs_{ci.pyomo_name}'

        # Groupby keys: rule_args (constraint index names)
        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        keys_repr = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        if term.param_name:
            pi = self.t.params[term.param_name]
            param_col  = pi.pyomo_name.lower()
            param_s    = f's_{pi.pyomo_name.lower()}'
            joined_df  = f'_df_{ci.pyomo_name}_w'
            # join on the dimension the param is indexed by so the 1-D series
            # aligns correctly onto a multi-level variable index
            on_dim = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
            self._emit(f"{joined_df} = {df_var}.join({param_s}, on='{on_dim}')")
            self._emit(
                f"{lhs_var} = ({joined_df}['{param_col}'] * {joined_df}['{term.var_name}']).groupby({keys_repr}).sum()"
            )
        else:
            self._emit(f"{lhs_var} = {df_var}.groupby({keys_repr})['{term.var_name}'].sum()")

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P2(self, ci: ConstrInfo):
        term = ci.lhs_terms[0]
        df_var = f'df_{term.var_name}'
        lhs_var = f'lhs_{ci.pyomo_name}'

        if term.param_name:
            pi = self.t.params[term.param_name]
            # Attach param as column
            self._emit(f"{df_var}['_{pi.pyomo_name}'] = pd.Series(data['{pi.data_key}'])")
            self._emit(
                f"{lhs_var} = ({df_var}['_{pi.pyomo_name}'] * {df_var}['{term.var_name}']).sum()"
            )
        else:
            self._emit(f"{lhs_var} = {df_var}['{term.var_name}'].sum()")

        rhs_repr = self._rhs_repr(ci)
        op_str = _py_op_str(ci.op)
        self._emit(f"m.addLConstr({lhs_var} {op_str} {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P3(self, ci: ConstrInfo):
        # flow_out - flow_in == rhs
        # Two SumTermInfo in lhs_terms: [flow_out_term, flow_in_term]
        assert len(ci.lhs_terms) == 2
        out_term, in_term = ci.lhs_terms[0], ci.lhs_terms[1]
        df_var = f'df_{out_term.var_name}'

        # For flow_out: groupby the fixed dims (rule_args that are NOT the loop var)
        # loop_var for out_term is e.g. 'j' (the destination)
        # loop_var for in_term is e.g. 'i' (the source)
        # All var index names:
        vi = self.t.vars[out_term.var_name]
        all_names = self.r.all_names_for_var(vi)

        # Determine constraint index names (what we groupby to)
        constr_names = []
        for s in ci.index_sets:
            constr_names.extend(self.r.names_for(s))

        # The loop vars appear in the variable's index — find which columns they map to
        out_loop = out_term.loop_var if isinstance(out_term.loop_var, str) else out_term.loop_var[0]
        in_loop = in_term.loop_var if isinstance(in_term.loop_var, str) else in_term.loop_var[0]

        # For flow_out: var is indexed as (source, dest, commodity)
        # The loop var is 'j' (dest) — we sum over it, keeping (source, commodity)
        # But the constraint uses 'node' for source — need to rename
        # We detect which rule_arg corresponds to source by looking at what's NOT the loop var

        # Build groupby keys for out: all var index names EXCEPT out_loop
        out_keys = [n for n in all_names if n != out_loop]
        in_keys = [n for n in all_names if n != in_loop]

        out_keys_repr = '[' + ', '.join(f"'{k}'" for k in out_keys) + ']'
        in_keys_repr = '[' + ', '.join(f"'{k}'" for k in in_keys) + ']'

        out_var = f'flow_out_{ci.pyomo_name}'
        in_var = f'flow_in_{ci.pyomo_name}'
        lhs_var = f'lhs_{ci.pyomo_name}'

        self._emit(f"{out_var} = {df_var}.groupby({out_keys_repr})['{out_term.var_name}'].sum()")
        self._emit(f"{in_var} = {df_var}.groupby({in_keys_repr})['{in_term.var_name}'].sum()")

        # Rename so both series share the same index names = constr_names
        # out_keys may differ from constr_names if variable index used different names
        rename_out = {k: c for k, c in zip(out_keys, constr_names) if k != c}
        rename_in = {k: c for k, c in zip(in_keys, constr_names) if k != c}
        if rename_out:
            self._emit(f"{out_var} = {out_var}.rename_axis(index={rename_out})")
        if rename_in:
            self._emit(f"{in_var} = {in_var}.rename_axis(index={rename_in})")

        self._emit(f"{lhs_var} = {out_var}.sub({in_var}, fill_value=0)")

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P4(self, ci: ConstrInfo):
        """Cross-dimensional merge (component balance / machine capacity)."""
        # Determine actual term list and whether sides are swapped
        if ci.lhs_is_direct_var:
            # LHS is direct var, RHS is the sum → swap: lhs=sum, rhs=var
            terms = ci.rhs_terms
            rhs_is_var = True
        else:
            terms = ci.lhs_terms
            rhs_is_var = False

        term = terms[0]
        pi = self.t.params[term.param_name] if term.param_name else None
        vi = self.t.vars[term.var_name]
        df_var = f'df_{term.var_name}'

        suffix = ci.pyomo_name
        lhs_var = f'lhs_{suffix}'

        if pi:
            param_s = f's_{pi.pyomo_name.lower()}'
            # flat names
            flat_param = f'df_{pi.pyomo_name.lower()}_flat_{suffix}'
            flat_var = f'df_{term.var_name}_flat_{suffix}'
            join_df = f'df_join_{suffix}'

            # Join key: the loop var (shared dimension between param and var)
            loop_v = term.loop_var
            if isinstance(loop_v, list):
                on_key = loop_v
                on_repr = '[' + ', '.join(f"'{k}'" for k in loop_v) + ']'
            else:
                on_key = [loop_v]
                on_repr = f"'{loop_v}'"

            self._emit(f"{flat_param} = {param_s}.reset_index()")
            self._emit(f"{flat_var} = {df_var}['{term.var_name}'].reset_index()")
            self._emit(f"{join_df} = pd.merge({flat_param}, {flat_var}, on={on_repr})")

            # groupby keys: constraint index names
            constr_names = []
            for s in ci.index_sets:
                constr_names.extend(self.r.names_for(s))
            keys_repr = '[' + ', '.join(f"{join_df}['{k}']" for k in constr_names) + ']'

            # param column name: the Series name or first non-join column
            param_col = pi.pyomo_name.lower()
            self._emit(
                f"{lhs_var} = ({join_df}['{param_col}'] * {join_df}['{term.var_name}']).groupby({keys_repr}).sum()"
            )
        else:
            # No param — pure sum over indexed set
            # Not typical but handle gracefully
            constr_names = []
            for s in ci.index_sets:
                constr_names.extend(self.r.names_for(s))
            keys_repr = '[' + ', '.join(f"'{k}'" for k in constr_names) + ']'
            self._emit(f"{lhs_var} = {df_var}.groupby({keys_repr})['{term.var_name}'].sum()")

        # Now determine RHS representation
        if rhs_is_var:
            # RHS is the direct var variable
            rhs_vi = self.t.vars[ci.lhs_direct_var]
            rhs_df = f'df_{ci.lhs_direct_var}'
            rhs_repr = f"{rhs_df}['{ci.lhs_direct_var}']"
            sense = _gurobi_sense(ci.op)
            self._emit(f"gppd.add_constrs(m, {rhs_repr}, {sense}, {lhs_var}, name='{ci.pyomo_name}')")
        else:
            # RHS is a param — check if lower-dim (needs reindex)
            rhs_repr = self._rhs_repr(ci)
            rhs_pi = self._rhs_param(ci)
            if rhs_pi and rhs_pi.index_sets:
                constr_names = []
                for s in ci.index_sets:
                    constr_names.extend(self.r.names_for(s))
                rhs_names = []
                for s in rhs_pi.index_sets:
                    rhs_names.extend(self.r.names_for(s))
                if len(rhs_names) < len(constr_names):
                    level_name = rhs_names[0]
                    aligned_var = f's_{rhs_pi.pyomo_name.lower()}_aligned_{suffix}'
                    self._emit(
                        f"{aligned_var} = {rhs_repr}.reindex({lhs_var}.index, level='{level_name}')"
                    )
                    rhs_repr = aligned_var
            sense = _gurobi_sense(ci.op)
            self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P5(self, ci: ConstrInfo):
        """Universal adjacency / indexed-relation engine.

        Works for any ∑_{j ∈ Relation[i]} x[...j...] pattern by:
          1. Flattening the indexed set dict into a relation DataFrame using
             registry-derived column names (not loop-var names).
          2. Renaming variable index columns only when a name conflict forces it.
          3. Reindexing the grouped result against the constraint's full index.
        """
        term = ci.lhs_terms[0]
        vi = self.t.vars[term.var_name]
        df_var = f'df_{term.var_name}'
        si = self.t.sets.get(term.iter_set)
        data_key = si.data_key if si else term.iter_set
        suffix = ci.pyomo_name

        # --- 1. Outer names ---
        # Full constraint outer names (used for groupby and reindex)
        full_outer_names = []
        for s in ci.index_sets:
            full_outer_names.extend(self.r.names_for(s))

        # Mapping outer names = registry names of the iter_set's parent set
        # (the set that indexes the relation, e.g. m.T indexes m.ValidStarts)
        parent_set_name = si.index_set if si else None
        if parent_set_name:
            map_outer_names = self.r.names_for(parent_set_name)
        else:
            map_outer_names = full_outer_names

        # --- 2. Inner names (registry-derived, with conflict resolution) ---
        var_index_names = self.r.all_names_for_var(vi)
        subscript_args = term.var_subscript_args   # e.g. ['s', 'tp'] or ['fac'] or ['orig', 'dest']
        loop_vars_flat = term.loop_var if isinstance(term.loop_var, list) else [term.loop_var]

        # Inner positions: subscript_args positions NOT in rule_args
        inner_registry = []   # registry name at each inner position
        inner_loop_vars = []  # corresponding loop var name
        for i, arg in enumerate(subscript_args):
            if arg not in ci.rule_args:
                inner_registry.append(var_index_names[i] if i < len(var_index_names) else arg)
                # find which loop var this is
                lv_idx = loop_vars_flat.index(arg) if arg in loop_vars_flat else len(inner_registry) - 1
                inner_loop_vars.append(loop_vars_flat[lv_idx] if lv_idx < len(loop_vars_flat) else arg)

        # Resolve naming conflicts: if inner registry name == outer name, use loop var name instead
        inner_col_names = []
        rename_map = {}  # var's registry col → col name to use in reset_index
        for reg, lv in zip(inner_registry, inner_loop_vars):
            if reg in full_outer_names:
                inner_col_names.append(lv)
                rename_map[reg] = lv
            else:
                inner_col_names.append(reg)

        # --- 3. Build relation mapping DataFrame ---
        map_df = f'df_map_{suffix}'
        map_col_names = map_outer_names + inner_col_names

        # Outer iteration variable(s) in the comprehension
        if len(map_outer_names) == 1:
            outer_var = map_outer_names[0]
        else:
            outer_var = '(' + ', '.join(map_outer_names) + ')'

        # Inner iteration variable(s) in the comprehension
        if len(inner_col_names) == 1:
            inner_var = inner_col_names[0]
        else:
            inner_var = '(' + ', '.join(inner_col_names) + ')'

        # Full row tuple in the comprehension
        row_parts = map_outer_names + inner_col_names
        row_tuple = '(' + ', '.join(row_parts) + ')'

        cols_repr = '[' + ', '.join(f"'{c}'" for c in map_col_names) + ']'
        self._emit(
            f"_mapping_{suffix} = [{row_tuple} for {outer_var}, _inner in data['{data_key}'].items() for {inner_var} in _inner]"
        )
        self._emit(f"{map_df} = pd.DataFrame(_mapping_{suffix}, columns={cols_repr})")

        # --- 4. Flatten variable; rename conflicting index columns ---
        reset_df = f'df_reset_{suffix}'
        if rename_map:
            rename_repr = '{' + ', '.join(f"'{k}': '{v}'" for k, v in rename_map.items()) + '}'
            self._emit(f"{reset_df} = {df_var}['{term.var_name}'].reset_index().rename(columns={rename_repr})")
        else:
            self._emit(f"{reset_df} = {df_var}['{term.var_name}'].reset_index()")

        # --- 5. Merge on inner column names ---
        lagged_df = f'df_lagged_{suffix}'
        if len(inner_col_names) == 1:
            on_repr = f"'{inner_col_names[0]}'"
        else:
            on_repr = '[' + ', '.join(f"'{c}'" for c in inner_col_names) + ']'
        self._emit(f"{lagged_df} = pd.merge({map_df}, {reset_df}, on={on_repr})")

        # --- 6. Groupby full outer names, sum ---
        lhs_var = f'lhs_{suffix}'
        keys_repr = '[' + ', '.join(f"'{k}'" for k in full_outer_names) + ']'
        self._emit(f"{lhs_var} = {lagged_df}.groupby({keys_repr})['{term.var_name}'].sum()")

        # --- 7. Reindex against constraint's full index (handles missing outer values) ---
        constr_idx = f'_idx_{suffix}'
        self._emit_constr_index(ci, constr_idx)
        self._emit(f"{lhs_var} = {lhs_var}.reindex({constr_idx}, fill_value=0.0)")

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    def _emit_constr_index(self, ci: ConstrInfo, idx_var: str):
        """Emit construction of the constraint's full index for reindexing."""
        if len(ci.index_sets) == 1:
            s = ci.index_sets[0]
            si = self.t.sets.get(s)
            key = si.data_key if si else s
            names = self.r.names_for(s)
            if len(names) == 1:
                self._emit(f"{idx_var} = pd.Index(data['{key}'], name='{names[0]}')")
            else:
                names_repr = str(names)
                self._emit(f"{idx_var} = pd.MultiIndex.from_tuples(data['{key}'], names={names_repr})")
        else:
            sets_repr = '[' + ', '.join(
                f"data['{(self.t.sets.get(s) or SetInfo('', '')).data_key or s}']"
                for s in ci.index_sets
            ) + ']'
            names = []
            for s in ci.index_sets:
                names.extend(self.r.names_for(s))
            names_repr = str(names)
            self._emit(f"{idx_var} = pd.MultiIndex.from_product({sets_repr}, names={names_repr})")

    # ------------------------------------------------------------------
    def _gen_P6(self, ci: ConstrInfo):
        """Direct var access, with optional subset filter."""
        var_name = ci.lhs_direct_var
        vi = self.t.vars.get(var_name, list(self.t.vars.values())[0])
        df_var = f'df_{var_name}'

        # Check for subset set
        subset_set = None
        for s in ci.index_sets:
            si = self.t.sets.get(s)
            if si and si.is_subset:
                subset_set = si
                break

        if subset_set:
            idx_var = f'{ci.pyomo_name}_idx'
            self._emit(f"{idx_var} = pd.IndexSlice[data['{subset_set.data_key}'], :]")
            lhs_repr = f"{df_var}.loc[{idx_var}, '{var_name}']"
        else:
            lhs_repr = f"{df_var}['{var_name}']"

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_repr}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _rhs_repr(self, ci: ConstrInfo) -> str:
        """Generate the RHS expression string."""
        rhs = ci.rhs_node
        if rhs is None:
            return '0'
        # m.Param[...] or m.Param (scalar attr access)
        if isinstance(rhs, ast.Attribute):
            attr = _node_is_m_attr(rhs)
            if attr and attr in self.t.params:
                return self._param_var_name(attr)
        if isinstance(rhs, ast.Subscript):
            attr = _node_is_m_attr(rhs.value)
            if attr and attr in self.t.params:
                return self._param_var_name(attr)
        # Constant
        if isinstance(rhs, ast.Constant):
            return repr(rhs.value)
        # Fallback: unparse
        try:
            return ast.unparse(rhs)
        except Exception:
            return '???'

    def _rhs_param(self, ci: ConstrInfo) -> Optional[ParamInfo]:
        rhs = ci.rhs_node
        if rhs is None:
            return None
        if isinstance(rhs, ast.Attribute):
            attr = _node_is_m_attr(rhs)
            if attr and attr in self.t.params:
                return self.t.params[attr]
        if isinstance(rhs, ast.Subscript):
            attr = _node_is_m_attr(rhs.value)
            if attr and attr in self.t.params:
                return self.t.params[attr]
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate(func) -> str:
    """
    Translate a restricted build_pyomo_model function into the source code
    of an equivalent build_vectorized_model function using gurobipy-pandas.

    Args:
        func: A Python function following Pyomo AbstractModel templatization rules.

    Returns:
        A string containing the complete source of build_vectorized_model.
    """
    raw_src = inspect.getsource(func)
    src = textwrap.dedent(raw_src)
    tree = ast.parse(src)
    func_def = tree.body[0]

    translator = _Translator(func_def)
    translator.parse()
    translator.classify()
    return translator.generate()
