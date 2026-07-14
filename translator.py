"""
Pyomo → Gurobi (COO/MVar) transpiler.

Reads a restricted ``build_pyomo_model(data)`` function as source text and
emits the source of an equivalent ``build_vectorized_model(data)`` that
constructs the same model directly through Gurobi's matrix API (sparse COO
matrices, ``addMVar`` + ``addMConstr``), plus an ``update_vectorized_model``
companion for in-place re-solves.  Pipeline: parse (AST) → classify constraint
shapes → reconcile index names → generate code.  See TRANSLATOR_GUIDE.md for
usage and authoring conventions; differential_test.py verifies output against
the Pyomo reference.

Public API:
    from translator import translate, solve, make_model_fn, solution_proxy
    code_str = translate(build_pyomo_model)     # function, source str, or
    fn = make_model_fn(src)                     #   exec'd function
    gp_model, values = solve(build_pyomo_model, data)
    sol = solution_proxy(values)                # sol.x[i, j].value
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
    parent_sets: list = field(default_factory=list)  # ALL positional parents, e.g. Set(m.D0, m.D15, ...)
    is_subset: bool = False        # declared with within=
    within_set: Optional[str] = None
    # When initialize= is a literal/comprehension rather than data['key'],
    # the unparsed expression; it is emitted verbatim in the generated code
    # (which has `data` in scope), e.g. "[(e, c) for e in ['H','L'] for c in data['D0']]".
    init_repr: Optional[str] = None


@dataclass
class VarInfo:
    pyomo_name: str
    index_sets: list                # ordered Pyomo set names
    vtype: str = 'CONTINUOUS'       # 'CONTINUOUS' | 'INTEGER' | 'BINARY'
    lb: object = None               # explicit lower bound from bounds=(lb, ub); None = default
    ub: object = None               # explicit upper bound from bounds=(lb, ub); None = +inf


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
    param_subscript_args: list = field(default_factory=list)  # args in m.Param[a, b]
    fixed_subscripts: list = field(default_factory=list)   # [(pos, literal)] e.g. m.z5['H', c]
    scalar_coeff: float = 1.0   # constant multiplier e.g. 0.1 * var
    # Intra-sum linear combination: all (var_name, param_name, subscript_args, sign) terms
    # sign is +1 or -1; populated when elt is any linear combination of vars
    intra_terms: list = field(default_factory=list)


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
    # P_affine: general affine row.  Each term is ('var', sign, coeff_expr,
    # var_name, subscript_args) or ('sum', sign, SumTermInfo); affine_rhs is a
    # model-free scalar expression string (may reference `data`).
    affine_terms: list = field(default_factory=list)
    affine_rhs: str = '0'


@dataclass
class ObjInfo:
    pyomo_name: str
    sense: str                     # 'MINIMIZE' | 'MAXIMIZE'
    lhs_terms: list = field(default_factory=list)
    signs: list = field(default_factory=list)  # +1 or -1 per term


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _node_is_data_method_call(node) -> Optional[str]:
    """If node is `list(data['key'].keys())` or similar, return 'key', else None.
    Handles list(data['key'].keys()), list(data['key'].values()), list(data['key'].items()).
    """
    if not (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ('list', 'sorted', 'tuple')):
        return None
    if not node.args:
        return None
    inner = node.args[0]
    if not (isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in ('keys', 'values', 'items')):
        return None
    return _node_is_data_subscript(inner.func.value)


def _extract_keyword(call_node, kw_name):
    for kw in call_node.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _extract_subscript_args(subscript_node: ast.Subscript) -> list:
    """Return identifier names from m.var[a, b, c] — used to map loop vars to index positions."""
    return _extract_subscript_parts(subscript_node)[0]


def _extract_subscript_parts(subscript_node: ast.Subscript) -> tuple:
    """Split m.var[a, "H", c] into (names, fixed): the identifier names in
    order, and [(position, literal_value)] for constant subscripts (a fixed
    slice of the variable, e.g. m.z5["H", c])."""
    sl = subscript_node.slice
    if isinstance(sl, ast.Index):   # Python < 3.9 wraps in ast.Index
        sl = sl.value
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    names, fixed = [], []
    for pos, e in enumerate(elts):
        if isinstance(e, ast.Name):
            names.append(e.id)
        elif isinstance(e, ast.Constant):
            fixed.append((pos, e.value))
    return names, fixed


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


def _collect_add_terms(node: ast.expr) -> list:
    """Recursively unwrap a left-associative BinOp(Add) tree into a flat list of leaf nodes."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _collect_add_terms(node.left) + _collect_add_terms(node.right)
    return [node]


def _collect_signed_terms(node: ast.expr, sign: int = 1) -> list:
    """Recursively collect (leaf_node, sign) pairs from a tree of Add/Sub BinOps.
    Handles arbitrary nesting: a + b - (c + d) → [(a,+1),(b,+1),(c,-1),(d,-1)]."""
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return (_collect_signed_terms(node.left, sign) +
                    _collect_signed_terms(node.right, sign))
        if isinstance(node.op, ast.Sub):
            return (_collect_signed_terms(node.left, sign) +
                    _collect_signed_terms(node.right, -sign))
    return [(node, sign)]


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
        init_repr = None
        init_kw = _extract_keyword(call, 'initialize')
        if init_kw is not None:
            data_key = (_node_is_data_subscript(init_kw)
                        or _node_is_data_method_call(init_kw))
            if data_key is None:
                # Literal / comprehension initializer: carry the expression
                # itself into the generated code (it may reference `data`).
                init_repr = ast.unparse(init_kw)
                data_key = name

        dimen = 1
        dimen_kw = _extract_keyword(call, 'dimen')
        if dimen_kw is not None and isinstance(dimen_kw, ast.Constant):
            dimen = dimen_kw.value

        # Positional args that are m.X: parent set(s) of an indexed set
        # (pyo.Set(m.D0, m.D15, ...) is indexed by the product D0 x D15).
        parent_sets = [a for a in (_node_is_m_attr(arg) for arg in call.args) if a]
        is_indexed = bool(parent_sets)
        index_set = parent_sets[0] if parent_sets else None

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
            parent_sets=parent_sets,
            is_subset=is_subset,
            within_set=within_set,
            init_repr=init_repr,
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

        # bounds=(lb, ub) -- a two-element tuple of numeric constants or None.
        lb = ub = None
        bounds_kw = _extract_keyword(call, 'bounds')
        if isinstance(bounds_kw, ast.Tuple) and len(bounds_kw.elts) == 2:
            def _const(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                        and isinstance(node.operand, ast.Constant):
                    return -float(node.operand.value)
                return None   # None literal or non-constant -> use solver default
            lb, ub = _const(bounds_kw.elts[0]), _const(bounds_kw.elts[1])

        self.vars[name] = VarInfo(pyomo_name=name, index_sets=index_sets, vtype=vtype,
                                  lb=lb, ub=ub)
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

        self.obj = ObjInfo(pyomo_name=name, sense=sense)

        rule_kw = _extract_keyword(call, 'rule')
        expr_kw  = _extract_keyword(call, 'expr')
        if rule_kw is not None:
            self.obj._rule_name = rule_kw.id if isinstance(rule_kw, ast.Name) else ''
        elif expr_kw is not None:
            self.obj._inline_expr = expr_kw

    # ------------------------------------------------------------------
    def classify(self):
        classifier = _RuleClassifier(self)
        for ci in self.constrs:
            if ci.inline_expr is not None:
                classifier.classify_inline(ci)
            elif ci.rule_name and ci.rule_name in self._rules:
                classifier.classify_constr(ci, self._rules[ci.rule_name])
        if self.obj:
            if hasattr(self.obj, '_rule_name'):
                rn = self.obj._rule_name
                if rn and rn in self._rules:
                    classifier.classify_obj(self.obj, self._rules[rn])
            elif hasattr(self.obj, '_inline_expr'):
                classifier.classify_obj_inline(self.obj, self.obj._inline_expr)

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

        # P_inter_add: BinOp(Add) of multiple independent sum() calls
        # e.g.  sum(x[p,s] for s in S) + sum(y[p,e] for e in E) <= Cap[p]
        if isinstance(lhs_node, ast.BinOp) and isinstance(lhs_node.op, ast.Add):
            add_nodes = _collect_add_terms(lhs_node)
            parsed = [self._parse_sum_call(n) for n in add_nodes]
            if all(t is not None for t in parsed):
                ci.pattern = 'P_inter_add'
                ci.lhs_terms = parsed
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
            # RHS contains variables (big-M rows, indicator linearizations,
            # var-vs-var bounds): general affine row.
            if self._node_has_var(rhs_node) and self._try_affine(ci, lhs_node, rhs_node, assigns):
                return
            # Otherwise P6 direct var
            ci.pattern = 'P6'
            ci.lhs_direct_var = lhs_m_var
            return

        # P1/P2/P4/P5/P_intra_add: LHS is a sum() call
        lhs_sum = self._parse_sum_call(lhs_node)
        if lhs_sum is not None:
            ci.lhs_terms.append(lhs_sum)
            if lhs_sum.intra_terms:
                # Intra-sum: sum(c1*x + c2*y - z for ...)
                ci.pattern = 'P_intra_add'
            elif lhs_sum.iter_is_indexed:
                if lhs_sum.param_name:
                    ci.pattern = 'P4'
                else:
                    ci.pattern = 'P5'
            elif not ci.rule_args:
                ci.pattern = 'P2'
            else:
                ci.pattern = 'P1'
            return

        # Fallback: general affine row, then P6
        if self._try_affine(ci, lhs_node, rhs_node, assigns):
            return
        ci.pattern = 'P6'

    def _node_has_var(self, node) -> bool:
        """True if the subtree references any m.<decision variable>."""
        for n in ast.walk(node):
            attr = _node_is_m_attr(n)
            if attr and attr in self.t.vars:
                return True
        return False

    def _node_model_free(self, node) -> bool:
        """True if the subtree references no model component at all (only
        literals and `data`), so it can be emitted verbatim as a scalar."""
        return not any(_node_is_m_attr(n) for n in ast.walk(node))

    def _try_affine(self, ci: ConstrInfo, lhs_node, rhs_node, assigns) -> bool:
        """General affine row:  a ± combination of direct variables, var-only
        sums, and model-free scalars on either side of the relation.  All
        variable terms are collected on the LHS with signs (preserving the
        written orientation, as Pyomo does); model-free leaves fold into a
        scalar RHS expression.  Populates ci and returns True only if every
        leaf is recognized."""
        # Pyomo normalizes `a >= b` by swapping into `b <= a`, so its reference
        # rows carry the swapped orientation; reproduce that here (committed to
        # ci.op only on success).
        swapped = ci.op == 'GEQ'
        if swapped:
            lhs_node, rhs_node = rhs_node, lhs_node
        leaves = (_collect_signed_terms(lhs_node, 1)
                  + _collect_signed_terms(rhs_node, -1))
        terms, const_parts = [], []
        for node, sign in leaves:
            if isinstance(node, ast.Name) and node.id in assigns:
                node = assigns[node.id]
            # var-only sum
            st = self._parse_sum_call(node)
            if st is not None:
                if st.param_name or st.intra_terms or st.scalar_coeff != 1.0:
                    return False
                terms.append(('sum', sign, st))
                continue
            # direct var, optionally scaled by a model-free coefficient
            coeff_expr, var_node = '1', None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                for side, other in ((node.left, node.right), (node.right, node.left)):
                    if (isinstance(side, ast.Subscript)
                            and (_node_is_m_attr(side.value) or '') in self.t.vars
                            and self._node_model_free(other)):
                        var_node, coeff_expr = side, ast.unparse(other)
                        break
            elif (isinstance(node, ast.Subscript)
                    and (_node_is_m_attr(node.value) or '') in self.t.vars):
                var_node = node
            if var_node is not None:
                vname = _node_is_m_attr(var_node.value)
                args, fixed = _extract_subscript_parts(var_node)
                if fixed or not set(args) <= set(ci.rule_args):
                    return False
                terms.append(('var', sign, coeff_expr, vname, args))
                continue
            # model-free scalar
            if self._node_model_free(node):
                const_parts.append((ast.unparse(node), sign))
                continue
            return False
        if not terms:
            return False
        if swapped:
            ci.op = 'LEQ'
        ci.pattern = 'P_affine'
        ci.affine_terms = terms
        # Leaves were moved to the LHS; constants go to the RHS with flipped sign.
        ci.affine_rhs = (' '.join(
            f"{'+' if -sign > 0 else '-'} ({expr})" for expr, sign in const_parts
        ) or '0')
        return True

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

        if isinstance(lhs_node, ast.BinOp) and isinstance(lhs_node.op, ast.Add):
            add_nodes = _collect_add_terms(lhs_node)
            parsed = [self._parse_sum_call(n) for n in add_nodes]
            if all(t is not None for t in parsed):
                ci.pattern = 'P_inter_add'
                ci.lhs_terms = parsed
                return

        # P4b/P6: direct var on LHS
        lhs_m_var = self._extract_direct_var(lhs_node)
        if lhs_m_var is not None:
            rhs_sum = self._parse_sum_call(ci.rhs_node)
            if rhs_sum is not None:
                ci.pattern = 'P4'
                ci.lhs_is_direct_var = True
                ci.lhs_direct_var = lhs_m_var
                ci.rhs_terms = [rhs_sum]
                return
            ci.pattern = 'P6'
            ci.lhs_direct_var = lhs_m_var
            return

        lhs_sum = self._parse_sum_call(lhs_node)
        if lhs_sum is not None:
            ci.lhs_terms.append(lhs_sum)
            if lhs_sum.intra_terms:
                ci.pattern = 'P_intra_add'
            elif lhs_sum.iter_is_indexed:
                ci.pattern = 'P4' if lhs_sum.param_name else 'P5'
            elif not ci.rule_args:
                ci.pattern = 'P2'
            else:
                ci.pattern = 'P1'
            return
        if self._try_affine(ci, lhs_node, ci.rhs_node, {}):
            return
        ci.pattern = 'P6'

    def classify_obj(self, obj: ObjInfo, rfunc: ast.FunctionDef):
        return_node = None
        for stmt in rfunc.body:
            if isinstance(stmt, ast.Return):
                return_node = stmt.value
        if return_node is None:
            return
        signed = self._collect_signed_sum_terms(return_node)
        obj.lhs_terms = [t for t, _s in signed]
        obj.signs = [s for _t, s in signed]

    def classify_obj_inline(self, obj: ObjInfo, expr_node: ast.expr):
        """Handle pyo.Objective(expr=...) — expression may be sum_A +/- sum_B."""
        signed = self._collect_signed_sum_terms(expr_node)
        obj.lhs_terms = [t for t, _s in signed]
        obj.signs = [s for _t, s in signed]

    def _collect_signed_sum_terms(self, node: ast.expr) -> list:
        """Collect (SumTermInfo, sign) pairs from a tree of Add/Sub sum() calls.
        sign is +1 for additive terms and -1 for subtracted terms."""
        result = []
        for leaf, sign in _collect_signed_terms(node):
            for term in self._parse_first_generator(leaf):
                result.append((term, sign))
        return result

    def _parse_first_generator(self, node: ast.expr) -> list:
        """Return a list with the first parseable SumTermInfo from a sum() call."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'sum'):
            return []
        if not node.args or not isinstance(node.args[0], ast.GeneratorExp):
            return []
        gen_exp = node.args[0]
        for comp in gen_exp.generators:
            term = self._parse_comprehension(gen_exp.elt, comp)
            if term is not None:
                return [term]
        return []

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

        intra_terms: list = []

        scalar_coeff_val = 1.0   # overridden in the Mult branch when a literal is present

        param_subscript_args: list = []
        fixed_subscripts: list = []

        if isinstance(elt, ast.Subscript):
            vn = _node_is_m_attr(elt.value)
            if vn and vn in self.t.vars:
                var_name = vn
                var_subscript_args, fixed_subscripts = _extract_subscript_parts(elt)
            elif vn and vn in self.t.params:
                param_name = vn
                param_subscript_args = _extract_subscript_args(elt)
        elif isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.Mult):
            # param * var  or  var * param  or  constant * var
            for side in [elt.left, elt.right]:
                if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)):
                    scalar_coeff_val = float(side.value)
                    continue
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
                    param_subscript_args = _extract_subscript_args(side)
        elif isinstance(elt, ast.BinOp) and isinstance(elt.op, (ast.Add, ast.Sub)):
            # General intra-sum linear combination: x + y - z, a*x + b*y, etc.
            intra_terms = self._parse_linear_comb(elt)
            if intra_terms:
                var_name, param_name, var_subscript_args, _ = intra_terms[0]

        if var_name is None:
            return None

        coeff = scalar_coeff_val
        return SumTermInfo(
            var_name=var_name,
            param_name=param_name,
            loop_var=loop_var,
            iter_set=iter_set,
            iter_is_indexed=iter_is_indexed,
            iter_index_arg=iter_index_arg,
            var_subscript_args=var_subscript_args,
            param_subscript_args=param_subscript_args,
            fixed_subscripts=fixed_subscripts,
            scalar_coeff=coeff,
            intra_terms=intra_terms,
        )

    def _extract_direct_var(self, node: ast.expr) -> Optional[str]:
        """If node is m.var[...] where var is a known Var, return var name."""
        if isinstance(node, ast.Subscript):
            m_attr = _node_is_m_attr(node.value)
            if m_attr and m_attr in self.t.vars:
                return m_attr
        return None

    def _parse_linear_comb(self, elt: ast.expr) -> list:
        """Parse a linear combination expression into a list of
        (var_name, param_name, subscript_args, sign) tuples.
        Handles arbitrary +/- nesting and param*var or var*param weighting."""
        result = []
        for node, sign in _collect_signed_terms(elt):
            if isinstance(node, ast.Subscript):
                attr = _node_is_m_attr(node.value)
                if attr and attr in self.t.vars:
                    result.append((attr, None, _extract_subscript_args(node), sign))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                vname, pname, sargs = None, None, []
                for side in [node.left, node.right]:
                    if not isinstance(side, ast.Subscript):
                        continue
                    attr = _node_is_m_attr(side.value)
                    if attr is None:
                        continue
                    if attr in self.t.vars:
                        vname = attr
                        sargs = _extract_subscript_args(side)
                    elif attr in self.t.params:
                        pname = attr
                if vname:
                    result.append((vname, pname, sargs, sign))
        return result


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
        # Pass 4: for param index sets that never appear anywhere else (e.g. a
        # sparse dimen=2 pair set used only as a Param domain), derive dimension
        # names from the param's subscript args in the rules (m.BOM[p, c] over
        # m.BOMPairs → BOMPairs: ['p', 'c']).
        self._register_param_dims_from_all_terms()

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
        # Scan the objective's terms too: a variable may appear only there
        # (its subscript args are all loop vars, since objectives take no
        # rule arguments beyond the model).
        pos_names: dict[str, dict[int, str]] = {}
        scan = list(self.t.constrs)
        if self.t.obj is not None and self.t.obj.lhs_terms:
            obj_ci = ConstrInfo(pyomo_name='__obj__', index_sets=[],
                                rule_name='', rule_args=[])
            obj_ci.lhs_terms = self.t.obj.lhs_terms
            scan.append(obj_ci)
        for ci in scan:
            rule_args_set = set(ci.rule_args)
            for term in ci.lhs_terms + ci.rhs_terms:
                # Register from the main term's subscript args
                if term.var_subscript_args:
                    vname = term.var_name
                    pos_names.setdefault(vname, {})
                    for pos, arg in enumerate(term.var_subscript_args):
                        if arg not in rule_args_set and pos not in pos_names[vname]:
                            pos_names[vname][pos] = arg
                # Register from intra_terms (general linear combination)
                for (ivname, _ipname, isargs, _sign) in term.intra_terms:
                    pos_names.setdefault(ivname, {})
                    for pos, arg in enumerate(isargs):
                        if arg not in rule_args_set and pos not in pos_names[ivname]:
                            pos_names[ivname][pos] = arg

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

    def _register_param_dims_from_all_terms(self):
        """Derive index-level names for param index sets from the subscript
        args used in the rules.  Positional, mirroring pass 3: the param's
        index sets are walked with a cursor over their dimens, and each
        unregistered set takes its slice of the subscript args."""
        for ci in self.t.constrs:
            for term in ci.lhs_terms + ci.rhs_terms:
                if not (term.param_name and term.param_subscript_args):
                    continue
                pi = self.t.params.get(term.param_name)
                if pi is None:
                    continue
                sargs = term.param_subscript_args
                cursor = 0
                for set_name in pi.index_sets:
                    si = self.t.sets.get(set_name)
                    dimen = si.dimen if si else 1
                    if set_name not in self.registry:
                        names = sargs[cursor:cursor + dimen]
                        if len(names) == dimen:
                            self.registry[set_name] = names
                    cursor += dimen

    def names_for(self, set_name: str) -> list[str]:
        if set_name in self.registry:
            return self.registry[set_name]
        # Fallback: synthesize names from the lowercase set name. The arity
        # must match the set's dimen for ANY dimen — pandas rejects a
        # MultiIndex whose names count differs from its level count.
        si = self.t.sets.get(set_name)
        dimen = (si.dimen or 1) if si else 1
        if dimen == 1:
            return [set_name.lower()]
        return [f'{set_name.lower()}_{k}' for k in range(dimen)]

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
        """Generate build_vectorized_model (and update_vectorized_model) source."""
        self.lines = []
        self._emit('def build_vectorized_model(data):', indent=0)
        self._emit('import gurobipy as gp')
        self._emit('import pandas as pd')
        self._emit('import numpy as np')
        self._emit('import scipy.sparse')
        self._emit('m = gp.Model()')
        self._emit('m._mconstr = {}')
        self._emit('m._rhs_ord = {}')
        self._emit('m._constr_idx = {}')
        self._emit('m._mvars = {}')
        self._emit('m._var_idx = {}')
        self._emit()

        self._gen_vars()
        self._gen_params()

        if self.t.obj:
            self._gen_objective()

        for ci in self.t.constrs:
            self._gen_constraint(ci)

        # Lazy value extractor attached to model (used by solve())
        extracts = ', '.join(
            f"'{vn}': pd.Series(_var_{vn}.X, index=_idx_{vn})"
            for vn in self.t._var_order
        )
        self._emit(f'm._get_values = lambda: {{{extracts}}}')
        self._emit('return m')

        self._gen_update_function()
        return '\n'.join(self.lines)

    # ------------------------------------------------------------------
    def _gen_vars(self):
        for vname in self.t._var_order:
            vi = self.t.vars[vname]
            self._emit_var(vi)
            self._emit()

    def _set_expr(self, set_name: str) -> str:
        """Expression for a plain set's elements in the generated code:
        data['key'] normally, or the verbatim initializer for sets declared
        with a literal/comprehension initialize= (which may reference data)."""
        si = self.t.sets.get(set_name)
        if si is None:
            return f"data[{set_name!r}]"
        if si.init_repr:
            return si.init_repr
        return f"data[{si.data_key!r}]"

    def _emit_var(self, vi: VarInfo):
        all_names = self.r.all_names_for_var(vi)

        has_tuple_set = any(
            (self.t.sets.get(s) or SetInfo('', '')).dimen > 1
            for s in vi.index_sets
        )

        idx_var  = f'_idx_{vi.pyomo_name}'
        var_obj  = f'_var_{vi.pyomo_name}'
        flat_var = f'_flat_{vi.pyomo_name}'
        names_repr = ', '.join(f"'{n}'" for n in all_names)

        if has_tuple_set:
            parts, loop_parts = [], []
            for s in vi.index_sets:
                si = self.t.sets.get(s)
                dimen = si.dimen if si else 1
                if dimen > 1:
                    ns = self.r.names_for(s)
                    loop_parts.append(f"({', '.join(ns)}) in {self._set_expr(s)}")
                    parts.extend(ns)
                else:
                    n = self.r.names_for(s)[0]
                    loop_parts.append(f"{n} in {self._set_expr(s)}")
                    parts.append(n)
            self._emit(f"{idx_var}_tuples = [({', '.join(parts)}) for {' for '.join(loop_parts)}]")
            self._emit(f"{idx_var} = pd.MultiIndex.from_tuples({idx_var}_tuples, names=[{names_repr}])")
        else:
            if len(vi.index_sets) == 1:
                self._emit(f"{idx_var} = pd.Index({self._set_expr(vi.index_sets[0])}, name='{all_names[0]}')")
            else:
                sets_repr = ', '.join(self._set_expr(s) for s in vi.index_sets)
                self._emit(f"{idx_var} = pd.MultiIndex.from_product([{sets_repr}], names=[{names_repr}])")

        # Lower/upper bounds: an explicit bounds=(lb, ub) overrides the domain
        # default (lb=0 for NonNegative*).  A None bound keeps the solver default
        # (lb 0.0, ub +inf).  Binary vars are already {0,1}; only honor bounds if
        # given (rare, but harmless).
        lb_repr = f"{vi.lb!r}" if vi.lb is not None else "0.0"
        ub_part = f", ub={vi.ub!r}" if vi.ub is not None else ""
        if vi.vtype == 'BINARY':
            self._emit(f"{var_obj} = m.addMVar(len({idx_var}), vtype=gp.GRB.BINARY, name='{vi.pyomo_name}')")
        elif vi.vtype == 'INTEGER':
            self._emit(f"{var_obj} = m.addMVar(len({idx_var}), lb={lb_repr}{ub_part}, vtype=gp.GRB.INTEGER, name='{vi.pyomo_name}')")
        else:
            self._emit(f"{var_obj} = m.addMVar(len({idx_var}), lb={lb_repr}{ub_part}, name='{vi.pyomo_name}')")

        self._emit(f"{flat_var} = pd.DataFrame({{'_col': np.arange(len({idx_var}))}}, index={idx_var}).reset_index()")
        self._emit(f"m._mvars['{vi.pyomo_name}'] = {var_obj}")
        self._emit(f"m._var_idx['{vi.pyomo_name}'] = {idx_var}")

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
    def _gen_objective(self, new_data: bool = False):
        """Emit the objective.

        new_data=True is used by _gen_update_function: substitutes 'data' with
        'new_data' and '_var_x' / '_idx_x' with 'm._mvars[x]' / 'm._var_idx[x]'.
        """
        obj = self.t.obj
        if not obj.lhs_terms:
            return
        signs = obj.signs if obj.signs else [1] * len(obj.lhs_terms)
        d = 'new_data' if new_data else 'data'
        self._emit('# Objective')
        obj_parts = []
        for i, (term, sign) in enumerate(zip(obj.lhs_terms, signs)):
            pi = self.t.params.get(term.param_name) if term.param_name else None
            vi = self.t.vars.get(term.var_name)
            idx_v = (f"m._var_idx['{term.var_name}']" if new_data
                     else f'_idx_{term.var_name}')
            var_v = (f"m._mvars['{term.var_name}']" if new_data
                     else f'_var_{term.var_name}')
            coeff = term.scalar_coeff
            c_var = f'_c_obj_t{i}'

            if pi and pi.index_sets:
                param_s = f's_{pi.pyomo_name.lower()}'
                param_col = pi.pyomo_name.lower()
                param_dims = []
                for ps in pi.index_sets:
                    param_dims.extend(self.r.names_for(ps))
                all_var_dims = self.r.all_names_for_var(vi) if vi else []
                # Re-extract param from new_data if needed
                if new_data:
                    si_pi = self.t.params[pi.pyomo_name]
                    ax = ('[' + ', '.join(f"'{n}'" for n in param_dims) + ']'
                          if len(param_dims) > 1 else f"'{param_dims[0]}'")
                    self._emit(f"{param_s} = pd.Series({d}['{si_pi.data_key}'], name='{param_col}').rename_axis({ax})")
                if sorted(param_dims) == sorted(all_var_dims):
                    self._emit(f"{c_var} = {param_s}.reindex({idx_v}).values")
                elif len(param_dims) == 1:
                    self._emit(f"{c_var} = {param_s}.reindex({idx_v}, level='{param_dims[0]}').values")
                else:
                    self._emit(f"{c_var} = {param_s}.reindex({idx_v}).values")
                if coeff != 1.0:
                    self._emit(f"{c_var} = {c_var} * {coeff}")
            elif pi and not pi.index_sets:
                # Scalar param
                scalar_val = pi.pyomo_name.lower()
                if new_data:
                    self._emit(f"{scalar_val} = {d}['{pi.data_key}']")
                val = f'{coeff} * {scalar_val}' if coeff != 1.0 else scalar_val
                self._emit(f"{c_var} = np.full(len({idx_v}), {val})")
            else:
                val = str(coeff) if coeff != 1.0 else '1.0'
                self._emit(f"{c_var} = np.full(len({idx_v}), {val})")

            # Fixed literal subscripts (e.g. m.z5["H", c]) select a slice of
            # the variable: mask the coefficient vector by an indicator on the
            # fixed level(s).
            for pos, lit in term.fixed_subscripts:
                all_names = self.r.all_names_for_var(vi) if vi else []
                if pos < len(all_names):
                    col = all_names[pos]
                    self._emit(f"{c_var} = {c_var} * "
                               f"({idx_v}.get_level_values('{col}') == {lit!r}).astype(float)")

            term_expr = f'{c_var} @ {var_v}'
            if not obj_parts:
                obj_parts.append(f'-{c_var} @ {var_v}' if sign < 0 else term_expr)
            else:
                obj_parts.append(f'- {c_var} @ {var_v}' if sign < 0 else f'+ {term_expr}')

        obj_expr = ' '.join(obj_parts)
        sense_str = f"gp.GRB.{'MINIMIZE' if obj.sense == 'MINIMIZE' else 'MAXIMIZE'}"
        self._emit(f"m.setObjective({obj_expr}, {sense_str})")
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
        elif ci.pattern == 'P_inter_add':
            self._gen_P_inter_add(ci)
        elif ci.pattern == 'P_intra_add':
            self._gen_P_intra_add(ci)
        elif ci.pattern == 'P_affine':
            self._gen_P_affine(ci)
        else:
            self._emit(f"# WARNING: unknown pattern for {ci.pyomo_name}")
        self._emit()

    # ------------------------------------------------------------------
    def _gen_P_affine(self, ci: ConstrInfo):
        """General affine row: one coefficient block per term (direct vars and
        var-only sums, signs and model-free coefficients baked into the block
        values), model-free scalar RHS.  Assembled with addConstr, preserving
        the written orientation so signs match the Pyomo reference exactly."""
        suffix = ci.pyomo_name
        idx_c = f'_idx_{suffix}'
        self._emit_constr_index(ci, idx_c)
        constr_df = f'_constr_{suffix}'
        self._emit(f"{constr_df} = pd.DataFrame({{'_row': np.arange(len({idx_c}))}}, "
                   f"index={idx_c}).reset_index()")
        n_repr = f'len({idx_c})'
        self._emit_constr_idx_store(ci, suffix, None)

        block_exprs = []
        for i, t in enumerate(ci.affine_terms):
            tsfx = f'{suffix}_t{i}'
            if t[0] == 'var':
                _, sign, coeff_expr, vname, args = t
                aligned = self._flat_var_aligned(vname, args, tsfx)
                dc = f'_dc_{tsfx}'
                on = '[' + ', '.join(f"'{a}'" for a in args) + ']'
                self._emit(f"{dc} = pd.merge({constr_df}, {aligned}, on={on})")
                vals = (f"np.full(len({dc}), {float(sign)})" if coeff_expr == '1'
                        else f"np.full(len({dc}), {float(sign)} * ({coeff_expr}))")
                self._emit(
                    f"_A_{tsfx} = scipy.sparse.csr_matrix("
                    f"({vals}, ({dc}['_row'].values, {dc}['_col'].values)), "
                    f"shape=({n_repr}, len(_idx_{vname})))"
                )
                block_exprs.append(f"_A_{tsfx} @ _var_{vname}")
            else:  # 'sum'
                _, sign, term = t
                if term.iter_is_indexed:
                    lagged, full_outer, _, _ = self._emit_indexed_relation_mapping(term, ci, tsfx)
                    ok = '[' + ', '.join(f"'{k}'" for k in full_outer) + ']'
                    coo = f'_coo_{tsfx}'
                    self._emit(f"{coo} = pd.merge({lagged}, {constr_df}, on={ok})[['_col', '_row']]")
                else:
                    filtered = self._flat_var_filtered(
                        term, self.t.vars[term.var_name], tsfx)
                    on_cols = [a for a in term.var_subscript_args if a in ci.rule_args]
                    ok = '[' + ', '.join(f"'{k}'" for k in on_cols) + ']'
                    coo = f'_coo_{tsfx}'
                    self._emit(f"{coo} = pd.merge({filtered}, {constr_df}, on={ok})")
                self._emit(
                    f"_A_{tsfx} = scipy.sparse.csr_matrix("
                    f"(np.full(len({coo}), {float(sign)}), "
                    f"({coo}['_row'].values, {coo}['_col'].values)), "
                    f"shape=({n_repr}, len(_idx_{term.var_name})))"
                )
                block_exprs.append(f"_A_{tsfx} @ _var_{term.var_name}")

        lhs_expr = ' + '.join(block_exprs)
        op_str = _py_op_str(ci.op)
        self._emit(
            f"m._mconstr['{ci.pyomo_name}'] = m.addConstr("
            f"({lhs_expr}) {op_str} np.full({n_repr}, {ci.affine_rhs}), "
            f"name='{ci.pyomo_name}')"
        )

    # ------------------------------------------------------------------
    def _emit_constr_rows_df(self, ci: ConstrInfo, suffix: str):
        """Emit constraint ordering DataFrame (_constr_{suffix}) and return (df_name, n_repr, rhs_s).

        The DataFrame has columns [*constr_index_names, '_row'] where _row is
        the ordinal row number in the constraint block's sparse matrix.
        Builds from the RHS param series only when its dimensions EXACTLY cover
        all constraint index dimensions; otherwise falls back to _emit_constr_index.
        """
        rhs_pi = self._rhs_param(ci)
        constr_df = f'_constr_{suffix}'
        if rhs_pi and rhs_pi.index_sets:
            rhs_s = f's_{rhs_pi.pyomo_name.lower()}'
            # Only use rhs_s if param dims cover ALL constraint dims
            param_dim_names = []
            for s in rhs_pi.index_sets:
                param_dim_names.extend(self.r.names_for(s))
            constr_dim_names = []
            for s in ci.index_sets:
                constr_dim_names.extend(self.r.names_for(s))
            if set(param_dim_names) == set(constr_dim_names):
                self._emit(
                    f"{constr_df} = pd.DataFrame({{'_row': np.arange(len({rhs_s}))}}, "
                    f"index={rhs_s}.index).reset_index()"
                )
                return constr_df, f'len({rhs_s})', rhs_s
        # Explicit build from constraint index sets
        idx_v = f'_idx_{suffix}'
        self._emit_constr_index(ci, idx_v)
        self._emit(
            f"{constr_df} = pd.DataFrame({{'_row': np.arange(len({idx_v}))}}, "
            f"index={idx_v}).reset_index()"
        )
        return constr_df, f'len({idx_v})', None

    def _b_repr_from_constr(self, ci: ConstrInfo, suffix: str, n_repr: str, rhs_s: Optional[str]) -> str:
        """Return the numpy b-vector expression for addMConstr."""
        if rhs_s:
            return f'{rhs_s}.values'
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and not rhs_pi.index_sets:
            return f'np.full({n_repr}, {rhs_pi.pyomo_name.lower()})'
        if rhs_pi and rhs_pi.index_sets:
            # Lower-dim param: reindex onto the full constraint index
            param_dim_names = []
            for s in rhs_pi.index_sets:
                param_dim_names.extend(self.r.names_for(s))
            rhs_var = f's_{rhs_pi.pyomo_name.lower()}'
            idx_v   = f'_idx_{suffix}'
            if len(param_dim_names) == 1:
                return f'{rhs_var}.reindex({idx_v}, level=\'{param_dim_names[0]}\').values'
            levels_repr = '[' + ', '.join(f"'{n}'" for n in param_dim_names) + ']'
            return f'{rhs_var}.reindex({idx_v}, level={levels_repr}).values'
        rhs_repr = self._rhs_repr(ci)
        if isinstance(ci.rhs_node, ast.Constant):
            return f'np.full({n_repr}, {ci.rhs_node.value})'
        return f'np.full({n_repr}, {rhs_repr})'

    # ------------------------------------------------------------------
    def _has_matrix_coefficients(self, ci: ConstrInfo) -> bool:
        """True if any data parameter appears as a coefficient in the A matrix."""
        if ci.pattern in ('P1', 'P2', 'P_inter_add'):
            return any(t.param_name for t in ci.lhs_terms)
        if ci.pattern == 'P4':
            terms = ci.rhs_terms if ci.lhs_is_direct_var else ci.lhs_terms
            return any(t.param_name for t in terms)
        if ci.pattern == 'P_intra_add':
            return any(pname for _, pname, _, _ in ci.lhs_terms[0].intra_terms)
        return False  # P3, P5, P6 have structural-only (1/-1) A matrices

    def _emit_constr_idx_store(self, ci: ConstrInfo, suffix: str, rhs_s):
        """Emit m._constr_idx storage when RHS tracking alone is insufficient for rebuild."""
        if not self._has_matrix_coefficients(ci):
            return
        if rhs_s is None:
            self._emit(f"m._constr_idx['{ci.pyomo_name}'] = _idx_{suffix}")

    def _b_repr_update(self, ci: ConstrInfo, n_repr: str) -> str:
        """Return the b-vector expression for use inside update_vectorized_model."""
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and not rhs_pi.index_sets:
            return f'np.full({n_repr}, {rhs_pi.pyomo_name.lower()})'
        if rhs_pi and rhs_pi.index_sets:
            return f's_{rhs_pi.pyomo_name.lower()}.values'
        if ci.rhs_node is not None and isinstance(ci.rhs_node, ast.Constant):
            return f'np.full({n_repr}, {ci.rhs_node.value})'
        return f'np.zeros({n_repr})'   # structural zero (P4 rhs_is_var)

    # ------------------------------------------------------------------
    def _gen_P1(self, ci: ConstrInfo):
        term = ci.lhs_terms[0]
        flat_v   = self._flat_var_filtered(term, self.t.vars[term.var_name], ci.pyomo_name)
        var_obj  = f'_var_{term.var_name}'
        idx_v    = f'_idx_{term.var_name}'
        suffix   = ci.pyomo_name

        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        keys_repr = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)
        self._emit_constr_idx_store(ci, suffix, rhs_s)
        coo = f'_coo_{suffix}'

        if term.param_name:
            pi = self.t.params[term.param_name]
            param_s   = f's_{pi.pyomo_name.lower()}'
            param_key = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
            self._emit(f"{coo} = pd.merge({flat_v}, {constr_df}, on={keys_repr})")
            self._emit(f"{coo} = {coo}.assign(_val={coo}['{param_key}'].map({param_s}))")
            val_repr = f"{coo}['_val'].values"
        else:
            self._emit(f"{coo} = pd.merge({flat_v}, {constr_df}, on={keys_repr})")
            val_repr = f'np.ones(len({coo}))'

        self._emit(
            f"_A_{suffix} = scipy.sparse.csr_matrix("
            f"({val_repr}, ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )
        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}, {var_obj}, {sense}, {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    # ------------------------------------------------------------------
    def _flat_var_filtered(self, term: SumTermInfo, vi: VarInfo, suffix: str) -> str:
        """Expression for ``_flat_{var}`` restricted to the columns whose index
        lies in the summation's iteration set.

        A scalar sum such as ``sum(u[p,q] for (p,q) in C)`` ranges the variable
        ``u`` (indexed by ``A``) over a *different* plain set ``C`` that selects a
        subset of its columns.  When the variable is indexed exactly by the
        iteration set, no restriction is needed and ``_flat_{var}`` is returned
        unchanged, preserving prior behavior.
        """
        flat = f'_flat_{term.var_name}'
        iter_set = term.iter_set
        if iter_set in vi.index_sets:
            return flat                      # iteration covers a whole index block
        si = self.t.sets.get(iter_set)
        if si is None or si.is_indexed:
            return flat                      # indexed relations are handled elsewhere
        # Filter on the loop-var columns only: for sum(z0[e,c,a,b] for (a,b) in D5)
        # inside a constraint over (e,c), the subset D5 restricts columns (a, b).
        # Use the CANONICAL column names of the flat frame at those positions
        # (the rule's local loop names may differ from the registry names).
        loop = term.loop_var if isinstance(term.loop_var, list) else [term.loop_var]
        all_names = self.r.all_names_for_var(vi)
        cols_list = [all_names[i] for i, a in enumerate(term.var_subscript_args)
                     if a in loop and i < len(all_names)] or all_names
        if (si.dimen or 1) != len(cols_list):
            return flat                      # arity mismatch: leave untouched
        cols = '[' + ', '.join(f"'{n}'" for n in cols_list) + ']'
        sel = f'_sel_{suffix}'
        self._emit(f"{sel} = pd.DataFrame(list({self._set_expr(iter_set)}), columns={cols})")
        filtered = f'{flat}_{suffix}'
        self._emit(f"{filtered} = pd.merge({flat}, {sel}, on={cols})")
        return filtered

    def _gen_P2(self, ci: ConstrInfo):
        """Scalar (single-row) constraint — 1×n sparse matrix."""
        term = ci.lhs_terms[0]
        vi      = self.t.vars[term.var_name]
        flat_v  = self._flat_var_filtered(term, vi, ci.pyomo_name)
        var_obj = f'_var_{term.var_name}'
        idx_v   = f'_idx_{term.var_name}'
        suffix  = ci.pyomo_name

        if term.param_name:
            pi = self.t.params[term.param_name]
            param_s   = f's_{pi.pyomo_name.lower()}'
            param_col = pi.pyomo_name.lower()
            coo = f'_coo_{suffix}'
            param_keys = []
            for s in pi.index_sets:
                param_keys.extend(self.r.names_for(s))
            if len(param_keys) == 1:
                self._emit(f"{coo} = {flat_v}.assign(_val={flat_v}['{param_keys[0]}'].map({param_s}))")
            else:
                on_repr = '[' + ', '.join(f"'{k}'" for k in param_keys) + ']'
                self._emit(f"{coo} = pd.merge({flat_v}, {param_s}.reset_index(), on={on_repr})")
                self._emit(f"{coo} = {coo}.rename(columns={{'{param_col}': '_val'}})")
            val_repr = f"{coo}['_val'].values"
            col_repr = f"{coo}['_col'].values"
            n_expr   = f'len({coo})'
        else:
            val_repr = f'np.ones(len({flat_v}))'
            col_repr = f"{flat_v}['_col'].values"
            n_expr   = f'len({flat_v})'

        self._emit(
            f"_A_{suffix} = scipy.sparse.csr_matrix("
            f"({val_repr}, (np.zeros({n_expr}, dtype=int), {col_repr})), "
            f"shape=(1, len({idx_v})))"
        )
        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}, {var_obj}, {sense}, np.array([{rhs_repr}]), name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P3(self, ci: ConstrInfo):
        assert len(ci.lhs_terms) == 2
        out_term, in_term = ci.lhs_terms[0], ci.lhs_terms[1]
        vi = self.t.vars[out_term.var_name]
        flat_v  = f'_flat_{out_term.var_name}'
        var_obj = f'_var_{out_term.var_name}'
        idx_v   = f'_idx_{out_term.var_name}'
        suffix  = ci.pyomo_name

        all_names = self.r.all_names_for_var(vi)
        constr_names = []
        for s in ci.index_sets:
            constr_names.extend(self.r.names_for(s))

        out_loop = out_term.loop_var if isinstance(out_term.loop_var, str) else out_term.loop_var[0]
        in_loop  = in_term.loop_var  if isinstance(in_term.loop_var,  str) else in_term.loop_var[0]
        # Use position-based filtering so registry names and loop-var names don't have to match
        out_loop_pos = (out_term.var_subscript_args.index(out_loop)
                        if out_loop in out_term.var_subscript_args else None)
        in_loop_pos  = (in_term.var_subscript_args.index(in_loop)
                        if in_loop  in in_term.var_subscript_args  else None)
        out_keys = [n for i, n in enumerate(all_names) if i != out_loop_pos]
        in_keys  = [n for i, n in enumerate(all_names) if i != in_loop_pos]

        # Build constraint ordering from RHS param
        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)

        # Forward arcs: variable (out_keys match constr_names)
        rename_out = {k: c for k, c in zip(out_keys, constr_names) if k != c}
        rename_in  = {k: c for k, c in zip(in_keys,  constr_names) if k != c}
        fwd = f'_fwd_{suffix}'
        bwd = f'_bwd_{suffix}'

        if rename_out:
            inv_out = '{' + ', '.join(f"'{v}': '{k}'" for k, v in rename_out.items()) + '}'
            fwd_constr = f'_constr_fwd_{suffix}'
            self._emit(f"{fwd_constr} = {constr_df}.rename(columns={inv_out})")
            ok = '[' + ', '.join(f"'{k}'" for k in out_keys) + ']'
            self._emit(f"{fwd} = pd.merge({flat_v}, {fwd_constr}, on={ok})[['_col', '_row']]")
        else:
            ok = '[' + ', '.join(f"'{k}'" for k in out_keys) + ']'
            self._emit(f"{fwd} = pd.merge({flat_v}, {constr_df}, on={ok})[['_col', '_row']]")

        if rename_in:
            inv_in = '{' + ', '.join(f"'{v}': '{k}'" for k, v in rename_in.items()) + '}'
            bwd_constr = f'_constr_bwd_{suffix}'
            self._emit(f"{bwd_constr} = {constr_df}.rename(columns={inv_in})")
            ik = '[' + ', '.join(f"'{k}'" for k in in_keys) + ']'
            self._emit(f"{bwd} = pd.merge({flat_v}, {bwd_constr}, on={ik})[['_col', '_row']]")
        else:
            ik = '[' + ', '.join(f"'{k}'" for k in in_keys) + ']'
            self._emit(f"{bwd} = pd.merge({flat_v}, {constr_df}, on={ik})[['_col', '_row']]")

        coo = f'_coo_{suffix}'
        self._emit(f"{coo} = pd.concat([{fwd}.assign(_val=1.0), {bwd}.assign(_val=-1.0)], ignore_index=True)")
        self._emit(
            f"_A_{suffix} = scipy.sparse.csr_matrix("
            f"({coo}['_val'].values, ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )
        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}, {var_obj}, {sense}, {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    # ------------------------------------------------------------------
    def _emit_indexed_relation_mapping(self, term: SumTermInfo, ci: ConstrInfo, suffix: str):
        """Emit the relation-mapping DataFrame + merged flat-variable DataFrame.

        Shared by _gen_P4_indexed_rhs and _gen_P5.  Returns:
          (lagged_df, full_outer_names, map_outer_names, inner_col_names)
        """
        vi = self.t.vars[term.var_name]
        flat_v = f'_flat_{term.var_name}'
        si = self.t.sets.get(term.iter_set)
        data_key = si.data_key if si else term.iter_set

        full_outer_names = []
        for s in ci.index_sets:
            full_outer_names.extend(self.r.names_for(s))

        # Outer key columns of the relation dict.  Prefer the subscript names
        # actually used in the rule (m.D16[c, a] → ['c', 'a'], matching the
        # constraint columns by construction); fall back to the registry names
        # of ALL parent sets (a multi-set indexed Set has a tuple key).
        if term.iter_index_arg:
            ia = term.iter_index_arg
            map_outer_names = list(ia) if isinstance(ia, tuple) else [ia]
        elif si and si.parent_sets:
            map_outer_names = []
            for ps in si.parent_sets:
                map_outer_names.extend(self.r.names_for(ps))
        else:
            map_outer_names = full_outer_names

        var_index_names  = self.r.all_names_for_var(vi)
        subscript_args   = term.var_subscript_args
        loop_vars_flat   = term.loop_var if isinstance(term.loop_var, list) else [term.loop_var]

        inner_registry, inner_loop_vars = [], []
        for i, arg in enumerate(subscript_args):
            if arg not in ci.rule_args:
                inner_registry.append(var_index_names[i] if i < len(var_index_names) else arg)
                lv_idx = loop_vars_flat.index(arg) if arg in loop_vars_flat else len(inner_registry) - 1
                inner_loop_vars.append(loop_vars_flat[lv_idx] if lv_idx < len(loop_vars_flat) else arg)

        inner_col_names, rename_map = [], {}
        for reg, lv in zip(inner_registry, inner_loop_vars):
            inner_col_names.append(lv)
            if reg != lv:
                rename_map[reg] = lv

        # Build mapping DataFrame
        map_df = f'_map_{suffix}'
        all_lv = loop_vars_flat
        map_cols = map_outer_names + all_lv
        outer_v = ('(' + ', '.join(map_outer_names) + ')' if len(map_outer_names) > 1
                   else map_outer_names[0])
        inner_v = ('(' + ', '.join(all_lv) + ')' if len(all_lv) > 1 else all_lv[0])
        row_tup = '(' + ', '.join(map_outer_names + all_lv) + ')'
        cols_repr = '[' + ', '.join(f"'{c}'" for c in map_cols) + ']'
        self._emit(
            f"_mapping_{suffix} = [{row_tup} for {outer_v}, _inner "
            f"in data['{data_key}'].items() for {inner_v} in _inner]"
        )
        self._emit(f"{map_df} = pd.DataFrame(_mapping_{suffix}, columns={cols_repr})")

        # Flat variable (possibly renamed)
        reset_df = f'_reset_{suffix}'
        all_var_post = [rename_map.get(n, n) for n in var_index_names]
        if rename_map:
            rr = '{' + ', '.join(f"'{k}': '{v}'" for k, v in rename_map.items()) + '}'
            self._emit(f"{reset_df} = {flat_v}.rename(columns={rr})")
        else:
            reset_df = flat_v  # reuse directly

        # Merge
        extra_on = [n for n in map_outer_names if n in all_var_post]
        merge_on = inner_col_names + extra_on
        on_repr = (f"'{merge_on[0]}'" if len(merge_on) == 1
                   else '[' + ', '.join(f"'{c}'" for c in merge_on) + ']')
        lagged_df = f'_lagged_{suffix}'
        self._emit(f"{lagged_df} = pd.merge({map_df}, {reset_df}, on={on_repr})")
        return lagged_df, full_outer_names, map_outer_names, inner_col_names

    # ------------------------------------------------------------------
    def _flat_var_aligned(self, var_name: str, target_names: list, suffix: str) -> str:
        """Expression for ``_flat_{var}`` whose index columns are renamed to
        ``target_names`` position-wise.

        A direct-variable reference is merged on the *constraint's* index names,
        but ``_flat_{var}`` carries the variable's own registry names.  When the
        same variable is subscripted under different argument names in different
        rules (e.g. ``v[p,q]`` in one constraint and ``v[r,w]`` in another), the
        merge key must be reconciled.  Returns ``_flat_{var}`` unchanged when the
        names already coincide, preserving prior behavior for every other model.
        """
        flat = f'_flat_{var_name}'
        vi = self.t.vars.get(var_name)
        if vi is None:
            return flat
        canonical = self.r.all_names_for_var(vi)
        if len(canonical) != len(target_names):
            return flat
        rename = {c: t for c, t in zip(canonical, target_names) if c != t}
        if not rename:
            return flat
        aligned = f'{flat}_{suffix}'
        self._emit(f"{aligned} = {flat}.rename(columns={rename!r})")
        return aligned

    def _gen_P4(self, ci: ConstrInfo):
        """Cross-dimensional merge — COO + addMConstr."""
        if ci.lhs_is_direct_var:
            terms = ci.rhs_terms
            rhs_is_var = True
        else:
            terms = ci.lhs_terms
            rhs_is_var = False

        term = terms[0]
        pi = self.t.params[term.param_name] if term.param_name else None

        if rhs_is_var and term.iter_is_indexed and not pi:
            self._gen_P4_indexed_rhs(ci, term)
            return

        vi = self.t.vars[term.var_name]
        flat_v  = f'_flat_{term.var_name}'
        var_obj = f'_var_{term.var_name}'
        idx_v   = f'_idx_{term.var_name}'
        suffix  = ci.pyomo_name

        constr_names = []
        for s in ci.index_sets:
            constr_names.extend(self.r.names_for(s))
        ck = '[' + ', '.join(f"'{k}'" for k in constr_names) + ']'

        # Build constraint row ordering
        if rhs_is_var:
            # No param RHS — build from constraint index
            self._emit_constr_index(ci, f'_idx_{suffix}')
            constr_df = f'_constr_{suffix}'
            self._emit(f"{constr_df} = pd.DataFrame({{'_row': np.arange(len(_idx_{suffix}))}}, index=_idx_{suffix}).reset_index()")
            n_repr = f'len(_idx_{suffix})'
            rhs_s  = None
            self._emit_constr_idx_store(ci, suffix, rhs_s)
        else:
            constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)
            self._emit_constr_idx_store(ci, suffix, rhs_s)

        coo = f'_coo_{suffix}'
        if pi:
            param_s = f's_{pi.pyomo_name.lower()}'
            param_col = pi.pyomo_name.lower()
            var_index_names = self.r.all_names_for_var(vi)
            loop_v = term.loop_var
            sargs  = term.var_subscript_args
            # Position-based: map each loop var to the registry name at the same subscript position
            if isinstance(loop_v, list):
                on_key = [var_index_names[sargs.index(k)] if k in sargs else k for k in loop_v]
            else:
                on_key = [var_index_names[sargs.index(loop_v)] if loop_v in sargs else loop_v]
            on_repr = (f"'{on_key[0]}'" if len(on_key) == 1
                       else '[' + ', '.join(f"'{k}'" for k in on_key) + ']')
            flat_p = f'_fp_{suffix}'
            m1     = f'_m1_{suffix}'
            self._emit(f"{flat_p} = {param_s}.reset_index()")
            self._emit(f"{m1} = pd.merge({flat_p}, {flat_v}, on={on_repr})")
            self._emit(f"{coo} = pd.merge({m1}, {constr_df}, on={ck})")
            val_repr = f"{coo}['{param_col}'].values"
        else:
            self._emit(f"{coo} = pd.merge({flat_v}, {constr_df}, on={ck})")
            val_repr = f'np.ones(len({coo}))'

        self._emit(
            f"_A_sum_{suffix} = scipy.sparse.csr_matrix("
            f"({val_repr}, ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )

        if rhs_is_var:
            # Constraint: sum(...) == direct_var  →  A_sum @ var_sum - I @ direct_var = 0
            lhs_v  = ci.lhs_direct_var
            flat_d = f'_flat_{lhs_v}'
            var_d  = f'_var_{lhs_v}'
            idx_d  = f'_idx_{lhs_v}'
            dc_df  = f'_dc_{suffix}'
            flat_d_aligned = self._flat_var_aligned(lhs_v, constr_names, suffix)
            self._emit(f"{dc_df} = pd.merge({constr_df}, {flat_d_aligned}, on={ck})")
            self._emit(
                f"_A_dir_{suffix} = scipy.sparse.csr_matrix("
                f"(np.ones(len({dc_df})), ({dc_df}['_row'].values, {dc_df}['_col'].values)), "
                f"shape=({n_repr}, len({idx_d})))"
            )
            # Emit as (direct - sum) OP 0, matching Pyomo's canonical orientation
            # so that constraint sense and coefficient signs coincide exactly.
            op_str = _py_op_str(ci.op)
            self._emit(
                f"m._mconstr['{ci.pyomo_name}'] = m.addConstr("
                f"(_A_dir_{suffix} @ {var_d} - _A_sum_{suffix} @ {var_obj}) "
                f"{op_str} np.zeros({n_repr}), name='{ci.pyomo_name}')"
            )
        else:
            b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
            sense = _gurobi_sense(ci.op)
            self._emit(
                f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr("
                f"_A_sum_{suffix}, {var_obj}, {sense}, {b}, name='{ci.pyomo_name}')"
            )
            if rhs_s:
                self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    # ------------------------------------------------------------------
    def _gen_P4_indexed_rhs(self, ci: ConstrInfo, term: SumTermInfo):
        """var == sum(y[...] for ... in Rel[...]) — COO + addMConstr."""
        suffix = ci.pyomo_name
        vi     = self.t.vars[term.var_name]
        idx_v  = f'_idx_{term.var_name}'
        var_obj = f'_var_{term.var_name}'

        lagged_df, full_outer, _, _ = self._emit_indexed_relation_mapping(term, ci, suffix)

        # Build constraint ordering from constraint index
        self._emit_constr_index(ci, f'_idx_{suffix}')
        constr_df = f'_constr_{suffix}'
        self._emit(
            f"{constr_df} = pd.DataFrame({{'_row': np.arange(len(_idx_{suffix}))}}, "
            f"index=_idx_{suffix}).reset_index()"
        )
        n_repr = f'len(_idx_{suffix})'

        ok = '[' + ', '.join(f"'{k}'" for k in full_outer) + ']'
        coo = f'_coo_{suffix}'
        self._emit(f"{coo} = pd.merge({lagged_df}, {constr_df}, on={ok})[['_col', '_row']]")

        self._emit(
            f"_A_rhs_{suffix} = scipy.sparse.csr_matrix("
            f"(np.ones(len({coo})), ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )

        # Negative-identity block for the LHS direct variable
        lhs_v  = ci.lhs_direct_var
        flat_d = f'_flat_{lhs_v}'
        var_d  = f'_var_{lhs_v}'
        idx_d  = f'_idx_{lhs_v}'
        ck     = '[' + ', '.join(f"'{k}'" for k in full_outer) + ']'
        dc_df  = f'_dc_{suffix}'
        flat_d_aligned = self._flat_var_aligned(lhs_v, full_outer, suffix)
        self._emit(f"{dc_df} = pd.merge({constr_df}, {flat_d_aligned}, on={ck})")
        self._emit(
            f"_A_dir_{suffix} = scipy.sparse.csr_matrix("
            f"(np.ones(len({dc_df})), ({dc_df}['_row'].values, {dc_df}['_col'].values)), "
            f"shape=({n_repr}, len({idx_d})))"
        )
        # Emit as (direct - sum) OP 0, matching Pyomo's canonical orientation
        # so that constraint sense and coefficient signs coincide exactly.
        op_str = _py_op_str(ci.op)
        self._emit(
            f"m._mconstr['{ci.pyomo_name}'] = m.addConstr("
            f"(_A_dir_{suffix} @ {var_d} - _A_rhs_{suffix} @ {var_obj}) "
            f"{op_str} np.zeros({n_repr}), name='{ci.pyomo_name}')"
        )

    # ------------------------------------------------------------------
    def _gen_P5(self, ci: ConstrInfo):
        """Indexed-relation constraint — COO + addMConstr."""
        term = ci.lhs_terms[0]
        var_obj = f'_var_{term.var_name}'
        idx_v   = f'_idx_{term.var_name}'
        suffix  = ci.pyomo_name

        lagged_df, full_outer, _, _ = self._emit_indexed_relation_mapping(term, ci, suffix)

        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)

        ok = '[' + ', '.join(f"'{k}'" for k in full_outer) + ']'
        coo = f'_coo_{suffix}'
        self._emit(f"{coo} = pd.merge({lagged_df}, {constr_df}, on={ok})")
        self._emit(
            f"_A_{suffix} = scipy.sparse.csr_matrix("
            f"(np.ones(len({coo})), ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )
        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}, {var_obj}, {sense}, {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    def _emit_constr_index(self, ci: ConstrInfo, idx_var: str):
        """Emit construction of the constraint's full index for reindexing."""
        if len(ci.index_sets) == 1:
            s = ci.index_sets[0]
            si = self.t.sets.get(s)
            names = self.r.names_for(s)
            if len(names) == 1:
                self._emit(f"{idx_var} = pd.Index({self._set_expr(s)}, name='{names[0]}')")
            else:
                names_repr = str(names)
                self._emit(f"{idx_var} = pd.MultiIndex.from_tuples({self._set_expr(s)}, names={names_repr})")
        else:
            names = []
            for s in ci.index_sets:
                names.extend(self.r.names_for(s))
            names_repr = str(names)
            has_tuple_set = any(
                (self.t.sets.get(s) or SetInfo('', '')).dimen > 1
                for s in ci.index_sets
            )
            if has_tuple_set:
                # from_product would treat a tuple set as ONE level; enumerate
                # the product explicitly, unpacking dimen>1 sets (as _emit_var does).
                parts, loop_parts = [], []
                for s in ci.index_sets:
                    ns = self.r.names_for(s)
                    if len(ns) > 1:
                        loop_parts.append(f"({', '.join(ns)}) in {self._set_expr(s)}")
                    else:
                        loop_parts.append(f"{ns[0]} in {self._set_expr(s)}")
                    parts.extend(ns)
                self._emit(f"{idx_var} = pd.MultiIndex.from_tuples("
                           f"[({', '.join(parts)}) for {' for '.join(loop_parts)}], names={names_repr})")
            else:
                sets_repr = '[' + ', '.join(self._set_expr(s) for s in ci.index_sets) + ']'
                self._emit(f"{idx_var} = pd.MultiIndex.from_product({sets_repr}, names={names_repr})")

    # ------------------------------------------------------------------
    def _gen_P_inter_add(self, ci: ConstrInfo):
        """sum(x[...]) + sum(y[...]) — per-term COO blocks, hstack, addMConstr."""
        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        suffix = ci.pyomo_name
        ck = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)
        self._emit_constr_idx_store(ci, suffix, rhs_s)

        a_blocks = []
        var_list_parts = []
        for i, term in enumerate(ci.lhs_terms):
            flat_v  = f'_flat_{term.var_name}'
            idx_v   = f'_idx_{term.var_name}'
            var_obj = f'_var_{term.var_name}'
            coo_i   = f'_coo_{suffix}_t{i}'

            if term.param_name:
                pi = self.t.params[term.param_name]
                param_s   = f's_{pi.pyomo_name.lower()}'
                param_col = pi.pyomo_name.lower()
                param_key = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
                self._emit(f"{coo_i} = pd.merge({flat_v}, {constr_df}, on={ck})")
                self._emit(f"{coo_i} = {coo_i}.assign(_val={coo_i}['{param_key}'].map({param_s}))")
                val_repr = f"{coo_i}['_val'].values"
            else:
                self._emit(f"{coo_i} = pd.merge({flat_v}, {constr_df}, on={ck})")
                val_repr = f'np.ones(len({coo_i}))'

            a_block = f'_A_{suffix}_t{i}'
            self._emit(
                f"{a_block} = scipy.sparse.csr_matrix("
                f"({val_repr}, ({coo_i}['_row'].values, {coo_i}['_col'].values)), "
                f"shape=({n_repr}, len({idx_v})))"
            )
            a_blocks.append(a_block)
            var_list_parts.append(f'list({var_obj})')

        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        op_str = _py_op_str(ci.op)
        # Build lhs as sum of A_block @ mvar terms (avoids list(MVar) incompatibility)
        terms_iter = zip(a_blocks, [f'_var_{term.var_name}' for term in ci.lhs_terms])
        lhs_expr = ' + '.join(f'{a} @ {v}' for a, v in terms_iter)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addConstr(({lhs_expr}) {op_str} {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    # ------------------------------------------------------------------
    def _gen_P_intra_add(self, ci: ConstrInfo):
        """sum(c1*x + c2*y - z for ...) — per-subterm COO blocks, hstack, addMConstr."""
        term0  = ci.lhs_terms[0]
        intra  = term0.intra_terms   # [(var_name, param_name, subscript_args, sign), ...]
        suffix = ci.pyomo_name

        outer_names = []
        for s in ci.index_sets:
            outer_names.extend(self.r.names_for(s))

        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)
        self._emit_constr_idx_store(ci, suffix, rhs_s)

        # Step 1: rename _col → _col_t{i} in a copy of each variable's flat DataFrame
        for i, (vname, _pname, _sargs, _sign) in enumerate(intra):
            fi = f'_fi{i}_{suffix}'
            self._emit(f"{fi} = _flat_{vname}.rename(columns={{'_col': '_col_t{i}'}})")

        # Step 2: iteratively merge flat DataFrames on shared index columns
        merged = f'_mi_{suffix}'
        self._emit(f"{merged} = _fi0_{suffix}")
        accumulated = set(self.r.all_names_for_var(self.t.vars[intra[0][0]]))
        for i, (vname, _pname, _sargs, _sign) in enumerate(intra[1:], 1):
            var_idx = set(self.r.all_names_for_var(self.t.vars[vname]))
            common  = sorted(accumulated & var_idx)
            if common:
                on_repr = (f"'{common[0]}'" if len(common) == 1
                           else '[' + ', '.join(f"'{c}'" for c in common) + ']')
                self._emit(f"{merged} = pd.merge({merged}, _fi{i}_{suffix}, on={on_repr})")
            else:
                self._emit(f"{merged} = {merged}.merge(_fi{i}_{suffix}, how='cross')")
            accumulated |= var_idx

        # Step 2b: merge any param Series into the merged DataFrame
        for i, (vname, pname, _sargs, _sign) in enumerate(intra):
            if pname:
                pi = self.t.params[pname]
                param_s   = f's_{pi.pyomo_name.lower()}'
                param_col = pi.pyomo_name.lower()
                pidx_names = []
                for ps in pi.index_sets:
                    pidx_names.extend(self.r.names_for(ps))
                on_repr = (f"'{pidx_names[0]}'" if len(pidx_names) == 1
                           else '[' + ', '.join(f"'{c}'" for c in pidx_names) + ']')
                self._emit(
                    f"{merged} = pd.merge({merged}, "
                    f"{param_s}.reset_index().rename(columns={{'{param_col}': '_pv{i}'}}), "
                    f"on={on_repr})"
                )

        # Step 3: merge with constr_df to get _row
        ck = '[' + ', '.join(f"'{k}'" for k in outer_names) + ']'
        self._emit(f"{merged} = pd.merge({merged}, {constr_df}, on={ck})")

        # Step 4: build A block and variable list per intra term
        a_blocks      = []
        var_list_parts = []
        for i, (vname, pname, _sargs, sign) in enumerate(intra):
            idx_v   = f'_idx_{vname}'
            var_obj = f'_var_{vname}'
            a_block = f'_A_{suffix}_t{i}'
            if pname:
                val_repr = (f"-{merged}['_pv{i}'].values" if sign < 0
                            else f"{merged}['_pv{i}'].values")
            else:
                val_repr = f'np.full(len({merged}), {float(sign)})'
            self._emit(
                f"{a_block} = scipy.sparse.csr_matrix("
                f"({val_repr}, ({merged}['_row'].values, {merged}['_col_t{i}'].values)), "
                f"shape=({n_repr}, len({idx_v})))"
            )
            a_blocks.append(a_block)
            var_list_parts.append(f'list({var_obj})')

        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        op_str = _py_op_str(ci.op)
        # Build lhs as sum of A_block @ mvar (sign baked into A coefficients)
        lhs_expr = ' + '.join(
            f'_A_{suffix}_t{i} @ _var_{vname}'
            for i, (vname, _pname, _sargs, _sign) in enumerate(intra)
        )
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addConstr(({lhs_expr}) {op_str} {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

    # ------------------------------------------------------------------
    def _gen_P6(self, ci: ConstrInfo):
        """Direct var access — COO + addMConstr."""
        var_name = ci.lhs_direct_var
        flat_v   = f'_flat_{var_name}'
        var_obj  = f'_var_{var_name}'
        idx_v    = f'_idx_{var_name}'
        suffix   = ci.pyomo_name

        constr_names = []
        for s in ci.index_sets:
            constr_names.extend(self.r.names_for(s))

        constr_df, n_repr, rhs_s = self._emit_constr_rows_df(ci, suffix)

        ck = '[' + ', '.join(f"'{k}'" for k in constr_names) + ']'
        coo = f'_coo_{suffix}'
        flat_v_aligned = self._flat_var_aligned(var_name, constr_names, suffix)
        self._emit(f"{coo} = pd.merge({constr_df}, {flat_v_aligned}, on={ck})")
        self._emit(
            f"_A_{suffix} = scipy.sparse.csr_matrix("
            f"(np.ones(len({coo})), ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len({idx_v})))"
        )
        b = self._b_repr_from_constr(ci, suffix, n_repr, rhs_s)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}, {var_obj}, {sense}, {b}, name='{ci.pyomo_name}')")
        if rhs_s:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = {rhs_s}.index")

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
        # Fallback: unparse the AST node as-is (requires Python 3.9+)
        return ast.unparse(rhs)

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

    # ------------------------------------------------------------------
    def _gen_update_function(self):
        """Emit update_vectorized_model(m, new_data) after the main function."""
        self._emit('', indent=0)
        self._emit('def update_vectorized_model(m, new_data):', indent=0)
        self._emit('import gurobipy as gp')
        self._emit('import pandas as pd')
        self._emit('import numpy as np')
        self._emit('import scipy.sparse')

        # Re-initialise all params from new_data
        for pname in self.t._param_order:
            pi = self.t.params[pname]
            var_name = f's_{pi.pyomo_name.lower()}'
            col_name = pi.pyomo_name.lower()
            if not pi.index_sets:
                self._emit(f"{pi.pyomo_name.lower()} = new_data['{pi.data_key}']")
            else:
                names = []
                for s in pi.index_sets:
                    names.extend(self.r.names_for(s))
                axis_repr = (f"'{names[0]}'" if len(names) == 1
                             else '[' + ', '.join(f"'{n}'" for n in names) + ']')
                self._emit(
                    f"{var_name} = pd.Series(new_data['{pi.data_key}'], "
                    f"name='{col_name}').rename_axis({axis_repr})"
                )
        if self.t.params:
            self._emit()

        for ci in self.t.constrs:
            if self._has_matrix_coefficients(ci):
                # Matrix coefficients changed — remove old constraint and rebuild
                self._emit(f"if '{ci.pyomo_name}' in m._mconstr:")
                self._emit(f"    m.remove(m._mconstr['{ci.pyomo_name}'])")
                self._emit_rebuild(ci)
            else:
                # Only RHS changed — hot-swap via setAttr (basis preserved)
                rhs_pi = self._rhs_param(ci)
                if not rhs_pi or not rhs_pi.index_sets:
                    continue
                rhs_s = f's_{rhs_pi.pyomo_name.lower()}'
                self._emit(f"if '{ci.pyomo_name}' in m._mconstr:")
                self._emit(
                    f"    m._mconstr['{ci.pyomo_name}'].setAttr("
                    f"'RHS', {rhs_s}.reindex(m._rhs_ord['{ci.pyomo_name}']).values)"
                )

        # Update objective
        if self.t.obj:
            self._emit()
            self._gen_objective(new_data=True)

        self._emit('return m')

    # ------------------------------------------------------------------
    def _constr_df_repr_update(self, ci: ConstrInfo) -> tuple:
        """Return (constr_df_expr, n_repr) for use inside update_vectorized_model."""
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and rhs_pi.index_sets:
            key = f"m._rhs_ord['{ci.pyomo_name}']"
            cd = f"pd.DataFrame({{'_row': np.arange(len({key}))}}, index={key}).reset_index()"
            return cd, f"len({key})"
        else:
            key = f"m._constr_idx['{ci.pyomo_name}']"
            cd = f"pd.DataFrame({{'_row': np.arange(len({key}))}}, index={key}).reset_index()"
            return cd, f"len({key})"

    def _flat_repr_update(self, var_name: str) -> str:
        """Emit and return the name of a reconstructed _flat frame from m._var_idx."""
        fname = f'_flat_{var_name}_u'
        self._emit(
            f"{fname} = pd.DataFrame({{'_col': np.arange(len(m._var_idx['{var_name}']))}}, "
            f"index=m._var_idx['{var_name}']).reset_index()"
        )
        return fname

    def _emit_rebuild(self, ci: ConstrInfo):
        """Dispatch to pattern-specific rebuild emitter."""
        if ci.pattern == 'P1':
            self._emit_rebuild_P1(ci)
        elif ci.pattern == 'P2':
            self._emit_rebuild_P2(ci)
        elif ci.pattern == 'P4':
            self._emit_rebuild_P4(ci)
        elif ci.pattern == 'P_inter_add':
            self._emit_rebuild_P_inter_add(ci)
        elif ci.pattern == 'P_intra_add':
            self._emit_rebuild_P_intra_add(ci)

    def _emit_rebuild_P1(self, ci: ConstrInfo):
        term   = ci.lhs_terms[0]
        var    = term.var_name
        suffix = ci.pyomo_name
        pi     = self.t.params[term.param_name]
        param_s   = f's_{pi.pyomo_name.lower()}'
        param_key = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        keys_repr = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        flat_u     = self._flat_repr_update(var)
        cd_expr, n_repr = self._constr_df_repr_update(ci)
        cdf = f'_cdf_{suffix}_u'
        coo = f'_coo_{suffix}_u'
        self._emit(f"{cdf} = {cd_expr}")
        self._emit(f"{coo} = pd.merge({flat_u}, {cdf}, on={keys_repr})")
        self._emit(f"{coo} = {coo}.assign(_val={coo}['{param_key}'].map({param_s}))")
        self._emit(
            f"_A_{suffix}_u = scipy.sparse.csr_matrix("
            f"({coo}['_val'].values, ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len(m._var_idx['{var}'])))"
        )
        b = self._b_repr_update(ci, n_repr)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}_u, m._mvars['{var}'], {sense}, {b}, name='{ci.pyomo_name}')")
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and rhs_pi.index_sets:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = s_{rhs_pi.pyomo_name.lower()}.index")

    def _emit_rebuild_P2(self, ci: ConstrInfo):
        term   = ci.lhs_terms[0]
        var    = term.var_name
        suffix = ci.pyomo_name
        pi     = self.t.params[term.param_name]
        param_s   = f's_{pi.pyomo_name.lower()}'
        param_col = pi.pyomo_name.lower()
        param_keys = []
        for s in pi.index_sets:
            param_keys.extend(self.r.names_for(s))

        flat_u = self._flat_repr_update(var)
        coo    = f'_coo_{suffix}_u'
        if len(param_keys) == 1:
            self._emit(f"{coo} = {flat_u}.assign(_val={flat_u}['{param_keys[0]}'].map({param_s}))")
        else:
            on_repr = '[' + ', '.join(f"'{k}'" for k in param_keys) + ']'
            self._emit(f"{coo} = pd.merge({flat_u}, {param_s}.reset_index(), on={on_repr})")
            self._emit(f"{coo} = {coo}.rename(columns={{'{param_col}': '_val'}})")
        self._emit(
            f"_A_{suffix}_u = scipy.sparse.csr_matrix("
            f"({coo}['_val'].values, (np.zeros(len({coo}), dtype=int), {coo}['_col'].values)), "
            f"shape=(1, len(m._var_idx['{var}'])))"
        )
        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr(_A_{suffix}_u, m._mvars['{var}'], {sense}, np.array([{rhs_repr}]), name='{ci.pyomo_name}')")

    def _emit_rebuild_P4(self, ci: ConstrInfo):
        if ci.lhs_is_direct_var:
            terms, rhs_is_var = ci.rhs_terms, True
        else:
            terms, rhs_is_var = ci.lhs_terms, False

        term   = terms[0]
        var    = term.var_name
        suffix = ci.pyomo_name
        pi     = self.t.params[term.param_name] if term.param_name else None

        constr_names = []
        for s in ci.index_sets:
            constr_names.extend(self.r.names_for(s))
        ck = '[' + ', '.join(f"'{k}'" for k in constr_names) + ']'

        cd_expr, n_repr = self._constr_df_repr_update(ci)
        cdf = f'_cdf_{suffix}_u'
        coo = f'_coo_{suffix}_u'
        self._emit(f"{cdf} = {cd_expr}")
        flat_u = self._flat_repr_update(var)

        if pi:
            param_s   = f's_{pi.pyomo_name.lower()}'
            param_col = pi.pyomo_name.lower()
            vi = self.t.vars[var]
            var_index_names = self.r.all_names_for_var(vi)
            loop_v = term.loop_var
            sargs  = term.var_subscript_args
            if isinstance(loop_v, list):
                on_key = [var_index_names[sargs.index(k)] if k in sargs else k for k in loop_v]
            else:
                on_key = [var_index_names[sargs.index(loop_v)] if loop_v in sargs else loop_v]
            on_repr = (f"'{on_key[0]}'" if len(on_key) == 1
                       else '[' + ', '.join(f"'{k}'" for k in on_key) + ']')
            fp  = f'_fp_{suffix}_u'
            m1  = f'_m1_{suffix}_u'
            self._emit(f"{fp} = {param_s}.reset_index()")
            self._emit(f"{m1} = pd.merge({fp}, {flat_u}, on={on_repr})")
            self._emit(f"{coo} = pd.merge({m1}, {cdf}, on={ck})")
            val_repr = f"{coo}['{param_col}'].values"
        else:
            self._emit(f"{coo} = pd.merge({flat_u}, {cdf}, on={ck})")
            val_repr = f'np.ones(len({coo}))'

        self._emit(
            f"_A_sum_{suffix}_u = scipy.sparse.csr_matrix("
            f"({val_repr}, ({coo}['_row'].values, {coo}['_col'].values)), "
            f"shape=({n_repr}, len(m._var_idx['{var}'])))"
        )

        if rhs_is_var:
            lhs_v  = ci.lhs_direct_var
            flat_d = self._flat_repr_update(lhs_v)
            dc     = f'_dc_{suffix}_u'
            self._emit(f"{dc} = pd.merge({cdf}, {flat_d}, on={ck})")
            self._emit(
                f"_A_neg_{suffix}_u = scipy.sparse.csr_matrix("
                f"(-np.ones(len({dc})), ({dc}['_row'].values, {dc}['_col'].values)), "
                f"shape=({n_repr}, len(m._var_idx['{lhs_v}'])))"
            )
            op_str = _py_op_str(ci.op)
            self._emit(
                f"m._mconstr['{ci.pyomo_name}'] = m.addConstr("
                f"(_A_sum_{suffix}_u @ m._mvars['{var}'] + _A_neg_{suffix}_u @ m._mvars['{lhs_v}']) "
                f"{op_str} np.zeros({n_repr}), name='{ci.pyomo_name}')"
            )
        else:
            b = self._b_repr_update(ci, n_repr)
            sense = _gurobi_sense(ci.op)
            self._emit(
                f"m._mconstr['{ci.pyomo_name}'] = m.addMConstr("
                f"_A_sum_{suffix}_u, m._mvars['{var}'], {sense}, {b}, name='{ci.pyomo_name}')"
            )
            rhs_pi = self._rhs_param(ci)
            if rhs_pi and rhs_pi.index_sets:
                self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = s_{rhs_pi.pyomo_name.lower()}.index")

    def _emit_rebuild_P_inter_add(self, ci: ConstrInfo):
        suffix = ci.pyomo_name
        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        ck = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        cd_expr, n_repr = self._constr_df_repr_update(ci)
        cdf = f'_cdf_{suffix}_u'
        self._emit(f"{cdf} = {cd_expr}")

        a_blocks = []
        for i, term in enumerate(ci.lhs_terms):
            var   = term.var_name
            flat_u = self._flat_repr_update(var)
            coo_i  = f'_coo_{suffix}_t{i}_u'
            if term.param_name:
                pi        = self.t.params[term.param_name]
                param_s   = f's_{pi.pyomo_name.lower()}'
                param_key = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
                self._emit(f"{coo_i} = pd.merge({flat_u}, {cdf}, on={ck})")
                self._emit(f"{coo_i} = {coo_i}.assign(_val={coo_i}['{param_key}'].map({param_s}))")
                val_repr = f"{coo_i}['_val'].values"
            else:
                self._emit(f"{coo_i} = pd.merge({flat_u}, {cdf}, on={ck})")
                val_repr = f'np.ones(len({coo_i}))'
            a_block = f'_A_{suffix}_t{i}_u'
            self._emit(
                f"{a_block} = scipy.sparse.csr_matrix("
                f"({val_repr}, ({coo_i}['_row'].values, {coo_i}['_col'].values)), "
                f"shape=({n_repr}, len(m._var_idx['{var}'])))"
            )
            a_blocks.append((a_block, var))

        b = self._b_repr_update(ci, n_repr)
        op_str = _py_op_str(ci.op)
        lhs_expr = ' + '.join(f'{a} @ m._mvars[\'{v}\']' for a, v in a_blocks)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addConstr(({lhs_expr}) {op_str} {b}, name='{ci.pyomo_name}')")
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and rhs_pi.index_sets:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = s_{rhs_pi.pyomo_name.lower()}.index")

    def _emit_rebuild_P_intra_add(self, ci: ConstrInfo):
        term0  = ci.lhs_terms[0]
        intra  = term0.intra_terms
        suffix = ci.pyomo_name

        outer_names = []
        for s in ci.index_sets:
            outer_names.extend(self.r.names_for(s))
        ck = '[' + ', '.join(f"'{k}'" for k in outer_names) + ']'

        cd_expr, n_repr = self._constr_df_repr_update(ci)
        cdf = f'_cdf_{suffix}_u'
        self._emit(f"{cdf} = {cd_expr}")

        # Rename _col columns
        for i, (vname, _, _, _) in enumerate(intra):
            fi = f'_fi{i}_{suffix}_u'
            flat_u = self._flat_repr_update(vname)
            self._emit(f"{fi} = {flat_u}.rename(columns={{'_col': '_col_t{i}'}})")

        # Merge flat frames
        merged = f'_mi_{suffix}_u'
        self._emit(f"{merged} = _fi0_{suffix}_u")
        accumulated = set(self.r.all_names_for_var(self.t.vars[intra[0][0]]))
        for i, (vname, _, _, _) in enumerate(intra[1:], 1):
            var_idx  = set(self.r.all_names_for_var(self.t.vars[vname]))
            common   = sorted(accumulated & var_idx)
            on_repr  = (f"'{common[0]}'" if len(common) == 1
                        else '[' + ', '.join(f"'{c}'" for c in common) + ']') if common else None
            if on_repr:
                self._emit(f"{merged} = pd.merge({merged}, _fi{i}_{suffix}_u, on={on_repr})")
            else:
                self._emit(f"{merged} = {merged}.merge(_fi{i}_{suffix}_u, how='cross')")
            accumulated |= var_idx

        # Merge param Series
        for i, (vname, pname, _, _) in enumerate(intra):
            if pname:
                pi        = self.t.params[pname]
                param_s   = f's_{pi.pyomo_name.lower()}'
                param_col = pi.pyomo_name.lower()
                pidx_names = []
                for ps in pi.index_sets:
                    pidx_names.extend(self.r.names_for(ps))
                on_repr = (f"'{pidx_names[0]}'" if len(pidx_names) == 1
                           else '[' + ', '.join(f"'{c}'" for c in pidx_names) + ']')
                self._emit(
                    f"{merged} = pd.merge({merged}, "
                    f"{param_s}.reset_index().rename(columns={{'{param_col}': '_pv{i}'}}), "
                    f"on={on_repr})"
                )

        self._emit(f"{merged} = pd.merge({merged}, {cdf}, on={ck})")

        a_blocks = []
        for i, (vname, pname, _, sign) in enumerate(intra):
            a_block = f'_A_{suffix}_t{i}_u'
            if pname:
                val_repr = (f"-{merged}['_pv{i}'].values" if sign < 0
                            else f"{merged}['_pv{i}'].values")
            else:
                val_repr = f'np.full(len({merged}), {float(sign)})'
            self._emit(
                f"{a_block} = scipy.sparse.csr_matrix("
                f"({val_repr}, ({merged}['_row'].values, {merged}['_col_t{i}'].values)), "
                f"shape=({n_repr}, len(m._var_idx['{vname}'])))"
            )
            a_blocks.append((a_block, vname))

        b = self._b_repr_update(ci, n_repr)
        op_str = _py_op_str(ci.op)
        lhs_expr = ' + '.join(f'{a} @ m._mvars[\'{v}\']' for a, v in a_blocks)
        self._emit(f"m._mconstr['{ci.pyomo_name}'] = m.addConstr(({lhs_expr}) {op_str} {b}, name='{ci.pyomo_name}')")
        rhs_pi = self._rhs_param(ci)
        if rhs_pi and rhs_pi.index_sets:
            self._emit(f"m._rhs_ord['{ci.pyomo_name}'] = s_{rhs_pi.pyomo_name.lower()}.index")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _source_of(func) -> tuple[str, Optional[str]]:
    """Recover (source, function_name) for translate()'s input.

    Accepts, in order of preference:
      * a plain string of Python source (the function need not exist yet);
      * a function carrying ``__transpile_source__`` — set by ``make_model_fn``
        for dynamically exec'd functions, which have no file for
        ``inspect.getsource`` to read;
      * an ordinary file-backed function.
    """
    if isinstance(func, str):
        return func, None
    src = getattr(func, '__transpile_source__', None)
    if src is not None:
        return src, func.__name__
    return inspect.getsource(func), func.__name__


def _find_func_def(tree: ast.Module, name: Optional[str]) -> ast.FunctionDef:
    """Locate the model-building FunctionDef in a parsed module: the one with
    the given name if present, else the first FunctionDef (so a source string
    may carry imports or other statements around the function)."""
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not defs:
        raise ValueError("no function definition found in the provided source")
    if name:
        for n in defs:
            if n.name == name:
                return n
    return defs[0]


def make_model_fn(source: str, name: str = 'build_pyomo_model'):
    """Create a model-building function from a source string, ready for both
    execution and transpilation.

    ``exec``'d functions have no source file, so ``inspect.getsource`` — and
    therefore ``translate`` — cannot see their code. This helper execs the
    source and attaches it to the function as ``__transpile_source__``, which
    ``translate`` (and everything built on it: ``solve``,
    ``differential_test.verify``) reads in preference to the file system.

        >>> src = '''
        ... import pyomo.environ as pyo
        ... def build_pyomo_model(data):
        ...     m = pyo.ConcreteModel()
        ...     ...
        ...     return m
        ... '''
        >>> fn = make_model_fn(src)
        >>> code = translate(fn)          # works despite having no file
    """
    ns: dict = {}
    exec(compile(source, f"<model:{name}>", "exec"), ns)
    if name not in ns or not callable(ns[name]):
        raise ValueError(f"source does not define a function named {name!r}")
    fn = ns[name]
    fn.__transpile_source__ = source
    return fn


def translate(func) -> str:
    """
    Translate a restricted build_pyomo_model function into the source code
    of an equivalent build_vectorized_model function using Gurobi's matrix API.

    Args:
        func: A Python function following Pyomo AbstractModel templatization
            rules; or a string containing such a function's source; or a
            dynamically created function from ``make_model_fn``.

    Returns:
        A string containing the complete source of build_vectorized_model.
    """
    raw_src, fn_name = _source_of(func)
    src = textwrap.dedent(raw_src)
    tree = ast.parse(src)
    func_def = _find_func_def(tree, fn_name)

    translator = _Translator(func_def)
    translator.parse()
    translator.classify()
    return translator.generate()


def solve(func, data, silent: bool = True):
    """
    Translate func, build and optimize the vectorized Gurobi model, and return
    the solved variable values.

    Args:
        func:   A build_pyomo_model function following translator conventions.
        data:   The data dict passed to the model builder.
        silent: Suppress Gurobi output (default True).

    Returns:
        (model, values) where
            model  – the solved gp.Model
            values – dict {var_name: pd.Series(index → float)}
                     Empty dict if the model was infeasible / no solution found.
    """
    code = translate(func)   # accepts file-backed, exec'd, or source-string input

    ns = {}
    exec(code, ns)
    model = ns['build_vectorized_model'](data)

    if silent:
        model.setParam('OutputFlag', 0)
    model.optimize()

    if model.SolCount == 0:
        return model, {}

    values = model._get_values()
    return model, values


# ---------------------------------------------------------------------------
# Solution proxy — zero-cost alternative to populate_pyomo
# ---------------------------------------------------------------------------

class _VarData:
    """Single variable element: exposes .value like Pyomo's VarData."""
    __slots__ = ('value',)

    def __init__(self, v: float):
        self.value = v


class _VarProxy:
    """Mimics a Pyomo Var component: proxy.x[i, j].value works as expected."""

    def __init__(self, series):
        self._d = {idx: _VarData(val) for idx, val in series.items()}

    def __getitem__(self, key):
        return self._d[key]


class SolutionProxy:
    """Duck-type substitute for a populated Pyomo ConcreteModel.

    Created by solution_proxy(values); supports model.var_name[idx].value
    without building a single Pyomo constraint.
    """

    def __init__(self, values: dict):
        for name, series in values.items():
            setattr(self, name, _VarProxy(series))


def solution_proxy(values: dict) -> SolutionProxy:
    """Wrap solve() values in a zero-cost proxy that mimics Pyomo's .value syntax.

    Args:
        values: Dict returned by solve(): {var_name: pd.Series(index → float)}.

    Returns:
        A SolutionProxy whose attributes are per-variable proxies.
        Access: sol.x['i1', 'j1'].value  (same as pyo_model.x['i1', 'j1'].value)

    Example::

        gp_model, values = solve(build_pyomo_model, data)
        sol = solution_proxy(values)
        print(sol.x['i1', 'j1'].value)
    """
    return SolutionProxy(values)


def populate_pyomo(pyomo_model, values: dict) -> None:
    """Load solve() values into a Pyomo model's variables.

    Note: building a full Pyomo model just to call this is usually unnecessary.
    Prefer solution_proxy(values) instead — it gives the same .value interface
    at zero cost.

    Args:
        pyomo_model: A Pyomo ConcreteModel (already built).
        values:      Dict returned by solve(): {var_name: pd.Series(index → float)}.
    """
    for var_name, series in values.items():
        pyo_var = getattr(pyomo_model, var_name, None)
        if pyo_var is None:
            continue
        for idx, val in series.items():
            pyo_var[idx].set_value(val)
