"""
Pyomo → gurobipy-pandas translator.

Public API:
    from translator import translate, solve, solution_proxy
    code_str = translate(build_pyomo_model)
    gp_model, values = solve(build_pyomo_model, data)
    sol = solution_proxy(values)   # sol.x[i, j].value
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


@dataclass
class ObjInfo:
    pyomo_name: str
    sense: str                     # 'MINIMIZE' | 'MAXIMIZE'
    lhs_terms: list = field(default_factory=list)
    signs: list = field(default_factory=list)  # +1 or -1 per term


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
        init_kw = _extract_keyword(call, 'initialize')
        if init_kw is not None:
            data_key = (_node_is_data_subscript(init_kw)
                        or _node_is_data_method_call(init_kw)
                        or name)

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

        if isinstance(elt, ast.Subscript):
            vn = _node_is_m_attr(elt.value)
            if vn and vn in self.t.vars:
                var_name = vn
                var_subscript_args = _extract_subscript_args(elt)
            elif vn and vn in self.t.params:
                param_name = vn
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

    def generate(self, extended: bool = False) -> str:
        """Generate build_vectorized_model source.

        When extended=True the generated function returns (model, {var_name: series})
        instead of just model, so callers can extract solved variable values.
        """
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

        if extended:
            vars_repr = (
                '{' +
                ', '.join(f"'{vn}': df_{vn}['{vn}']" for vn in self.t._var_order) +
                '}'
            )
            self._emit(f'return m, {vars_repr}')
        else:
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
        signs = obj.signs if obj.signs else [1] * len(obj.lhs_terms)
        self._emit('# Objective')
        part_vars = []
        part_signs = []
        for i, (term, sign) in enumerate(zip(obj.lhs_terms, signs)):
            pi = self.t.params.get(term.param_name) if term.param_name else None
            vi = self.t.vars.get(term.var_name)
            df_var = f'df_{term.var_name}'
            part = f'_obj_t{i}'
            coeff = term.scalar_coeff
            if pi and pi.index_sets:
                param_col = pi.pyomo_name.lower()
                joined = f'df_{term.var_name}_obj_{i}'
                # Determine join style: full-index alignment vs broadcast
                param_dims = []
                for ps in pi.index_sets:
                    param_dims.extend(self.r.names_for(ps))
                all_var_dims = self.r.all_names_for_var(vi) if vi else []
                if sorted(param_dims) == sorted(all_var_dims):
                    self._emit(f"{joined} = {df_var}.join(s_{pi.pyomo_name.lower()})")
                elif len(param_dims) == 1:
                    self._emit(f"{joined} = {df_var}.join(s_{pi.pyomo_name.lower()}, on='{param_dims[0]}')")
                else:
                    on_repr = '[' + ', '.join(f"'{d}'" for d in param_dims) + ']'
                    self._emit(f"{joined} = {df_var}.join(s_{pi.pyomo_name.lower()}, on={on_repr})")
                inner = f"{joined}['{param_col}'] * {joined}['{term.var_name}']"
                inner = f"{coeff} * {inner}" if coeff != 1.0 else inner
                self._emit(f"{part} = ({inner}).sum()")
            else:
                inner = f"{df_var}['{term.var_name}']"
                inner = f"{coeff} * {inner}" if coeff != 1.0 else inner
                self._emit(f"{part} = ({inner}).sum()")
            part_vars.append(part)
            part_signs.append(sign)
        # Assemble signed objective expression
        obj_parts = []
        for pv, sign in zip(part_vars, part_signs):
            if not obj_parts:
                obj_parts.append(f"-{pv}" if sign < 0 else pv)
            else:
                obj_parts.append(f"- {pv}" if sign < 0 else f"+ {pv}")
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

        # When LHS is a direct var and RHS is an indexed-set sum (no param),
        # use the P5-style flatten+merge+groupby pipeline.
        if rhs_is_var and term.iter_is_indexed and not pi:
            self._gen_P4_indexed_rhs(ci, term)
            return

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

            # Join key: the loop var (shared dimension between param and var),
            # restricted to the variable's actual index columns (handles tuple projection).
            var_index_names = self.r.all_names_for_var(vi)
            loop_v = term.loop_var
            if isinstance(loop_v, list):
                on_key = [k for k in loop_v if k in var_index_names]
            else:
                on_key = [loop_v] if loop_v in var_index_names else [loop_v]
            if len(on_key) == 1:
                on_repr = f"'{on_key[0]}'"
            else:
                on_repr = '[' + ', '.join(f"'{k}'" for k in on_key) + ']'

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
            # No param — pure sum over indexed set (non-indexed case)
            constr_names = []
            for s in ci.index_sets:
                constr_names.extend(self.r.names_for(s))
            keys_repr = '[' + ', '.join(f"'{k}'" for k in constr_names) + ']'
            self._emit(f"{lhs_var} = {df_var}.groupby({keys_repr})['{term.var_name}'].sum()")

        # Now determine RHS representation
        if rhs_is_var:
            # RHS is the direct var variable
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
    def _gen_P4_indexed_rhs(self, ci: ConstrInfo, term: SumTermInfo):
        """Handle var == sum(y[...] for ... in Rel[...]) using P5-like pipeline.

        Called when LHS is a direct variable and RHS is a sum over an indexed
        relation set with no param weight — same flatten+merge+groupby as P5
        but the constraint is emitted with the direct var on the LHS.
        """
        vi = self.t.vars[term.var_name]
        df_var = f'df_{term.var_name}'
        si = self.t.sets.get(term.iter_set)
        data_key = si.data_key if si else term.iter_set
        suffix = ci.pyomo_name

        # Outer names (constraint index)
        full_outer_names = []
        for s in ci.index_sets:
            full_outer_names.extend(self.r.names_for(s))

        # Parent set of the relation (what the outer key indexes)
        parent_set_name = si.index_set if si else None
        if parent_set_name:
            map_outer_names = self.r.names_for(parent_set_name)
        else:
            map_outer_names = full_outer_names

        var_index_names = self.r.all_names_for_var(vi)
        subscript_args = term.var_subscript_args
        loop_vars_flat = term.loop_var if isinstance(term.loop_var, list) else [term.loop_var]

        # Inner col names (from subscript args not in rule args, mapped to registry names)
        inner_registry = []
        inner_loop_vars = []
        for i, arg in enumerate(subscript_args):
            if arg not in ci.rule_args:
                inner_registry.append(var_index_names[i] if i < len(var_index_names) else arg)
                lv_idx = loop_vars_flat.index(arg) if arg in loop_vars_flat else len(inner_registry) - 1
                inner_loop_vars.append(loop_vars_flat[lv_idx] if lv_idx < len(loop_vars_flat) else arg)

        inner_col_names = []
        rename_map = {}
        for reg, lv in zip(inner_registry, inner_loop_vars):
            inner_col_names.append(lv)
            if reg != lv:
                rename_map[reg] = lv

        # Build mapping DataFrame using all loop var fields (handles projection)
        map_df = f'df_map_{suffix}'
        all_loop_var_names = loop_vars_flat
        map_col_names = map_outer_names + all_loop_var_names

        if len(map_outer_names) == 1:
            outer_var = map_outer_names[0]
        else:
            outer_var = '(' + ', '.join(map_outer_names) + ')'

        if len(all_loop_var_names) == 1:
            inner_var = all_loop_var_names[0]
        else:
            inner_var = '(' + ', '.join(all_loop_var_names) + ')'

        row_parts = map_outer_names + all_loop_var_names
        row_tuple = '(' + ', '.join(row_parts) + ')'
        cols_repr = '[' + ', '.join(f"'{c}'" for c in map_col_names) + ']'

        self._emit(f"_mapping_{suffix} = [{row_tuple} for {outer_var}, _inner in data['{data_key}'].items() for {inner_var} in _inner]")
        self._emit(f"{map_df} = pd.DataFrame(_mapping_{suffix}, columns={cols_repr})")

        # Flatten variable; rename conflicting index columns
        reset_df = f'df_reset_{suffix}'
        if rename_map:
            rename_repr = '{' + ', '.join(f"'{k}': '{v}'" for k, v in rename_map.items()) + '}'
            self._emit(f"{reset_df} = {df_var}['{term.var_name}'].reset_index().rename(columns={rename_repr})")
        else:
            self._emit(f"{reset_df} = {df_var}['{term.var_name}'].reset_index()")

        # Merge on inner col names plus any outer index cols also in the variable
        lagged_df = f'df_lagged_{suffix}'
        all_var_names_post_rename = [rename_map.get(n, n) for n in var_index_names]
        extra_on = [n for n in map_outer_names if n in all_var_names_post_rename]
        merge_on = inner_col_names + extra_on
        if len(merge_on) == 1:
            on_repr = f"'{merge_on[0]}'"
        else:
            on_repr = '[' + ', '.join(f"'{c}'" for c in merge_on) + ']'
        self._emit(f"{lagged_df} = pd.merge({map_df}, {reset_df}, on={on_repr})")

        # Groupby outer names, sum
        rhs_sum_var = f'rhs_{suffix}'
        keys_repr = '[' + ', '.join(f"'{k}'" for k in full_outer_names) + ']'
        self._emit(f"{rhs_sum_var} = {lagged_df}.groupby({keys_repr})['{term.var_name}'].sum()")

        # Reindex against the LHS direct var's index
        lhs_df = f'df_{ci.lhs_direct_var}'
        self._emit(f"{rhs_sum_var} = {rhs_sum_var}.reindex({lhs_df}.index, fill_value=0.0)")

        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_df}['{ci.lhs_direct_var}'], {sense}, {rhs_sum_var}, name='{ci.pyomo_name}')")

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

        # Use loop var names as the column names in the mapping DataFrame so that:
        # 1. The comprehension correctly unpacks ALL loop var fields (projection support).
        # 2. The variable reset_index is renamed to match these names (merge alignment).
        # rename_map covers BOTH outer-name conflicts AND cases where registry name ≠ loop var name.
        inner_col_names = []
        rename_map = {}  # var's registry col → loop var col name to use after reset_index
        for reg, lv in zip(inner_registry, inner_loop_vars):
            inner_col_names.append(lv)       # always use loop var name as the column
            if reg != lv:
                rename_map[reg] = lv          # rename variable index col to match

        # --- 3. Build relation mapping DataFrame ---
        # Use ALL loop var fields for comprehension unpacking (handles projection:
        # e.g., loop (f,w,s) but var subscript [f,s] — must unpack all three).
        map_df = f'df_map_{suffix}'
        all_loop_var_names = loop_vars_flat  # full tuple, may be superset of inner_col_names
        map_col_names = map_outer_names + all_loop_var_names

        # Outer iteration variable(s) in the comprehension
        if len(map_outer_names) == 1:
            outer_var = map_outer_names[0]
        else:
            outer_var = '(' + ', '.join(map_outer_names) + ')'

        # Inner iteration variable(s) — unpack all loop var fields
        if len(all_loop_var_names) == 1:
            inner_var = all_loop_var_names[0]
        else:
            inner_var = '(' + ', '.join(all_loop_var_names) + ')'

        # Full row tuple in the comprehension
        row_parts = map_outer_names + all_loop_var_names
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

        # --- 5. Merge on inner column names (plus any outer-index columns that
        # also appear in the variable's reset_index, to avoid pandas duplicate-column
        # suffixes when the variable subscript contains the constraint's outer index,
        # e.g. x[i, j] iterated over SubSet[i]). ---
        lagged_df = f'df_lagged_{suffix}'
        all_var_names_post_rename = [rename_map.get(n, n) for n in var_index_names]
        extra_on = [n for n in map_outer_names if n in all_var_names_post_rename]
        merge_on = inner_col_names + extra_on
        if len(merge_on) == 1:
            on_repr = f"'{merge_on[0]}'"
        else:
            on_repr = '[' + ', '.join(f"'{c}'" for c in merge_on) + ']'
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
    def _gen_P_inter_add(self, ci: ConstrInfo):
        """sum(x[...]) + sum(y[...]) — combine separate groupby results with .add()."""
        groupby_keys = []
        for s in ci.index_sets:
            groupby_keys.extend(self.r.names_for(s))
        keys_repr = '[' + ', '.join(f"'{k}'" for k in groupby_keys) + ']'

        term_var_names = []
        for idx, term in enumerate(ci.lhs_terms):
            df_v = f'df_{term.var_name}'
            lhs_i = f'lhs_{ci.pyomo_name}_t{idx}'
            if term.param_name:
                pi = self.t.params[term.param_name]
                on_dim = self.r.names_for(pi.index_sets[0])[0] if pi.index_sets else None
                joined = f'_df_{ci.pyomo_name}_w{idx}'
                self._emit(f"{joined} = {df_v}.join(s_{pi.pyomo_name.lower()}, on='{on_dim}')")
                self._emit(
                    f"{lhs_i} = ({joined}['{pi.pyomo_name.lower()}'] * {joined}['{term.var_name}']).groupby({keys_repr}).sum()"
                )
            else:
                self._emit(f"{lhs_i} = {df_v}.groupby({keys_repr})['{term.var_name}'].sum()")
            term_var_names.append(lhs_i)

        lhs_var = f'lhs_{ci.pyomo_name}'
        # Chain .add(fill_value=0) across all terms
        result = term_var_names[0]
        for tv in term_var_names[1:]:
            self._emit(f"{lhs_var} = {result}.add({tv}, fill_value=0)")
            result = lhs_var
        if len(term_var_names) == 1:
            self._emit(f"{lhs_var} = {term_var_names[0]}")

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P_intra_add(self, ci: ConstrInfo):
        """General intra-sum linear combination: sum(c1*x + c2*y - z for ...) op rhs.
        Works for any number of variables with any compatible index shapes by
        flattening each term then iteratively merging on shared index columns."""
        term = ci.lhs_terms[0]
        intra = term.intra_terms   # [(var_name, param_name, subscript_args, sign), ...]
        suffix = ci.pyomo_name

        outer_names = []
        for s in ci.index_sets:
            outer_names.extend(self.r.names_for(s))
        keys_repr = (f"'{outer_names[0]}'" if len(outer_names) == 1
                     else '[' + ', '.join(f"'{k}'" for k in outer_names) + ']')

        # --- Step 1: flatten each variable (+ its param coefficient if present) ---
        flat_infos = []
        for i, (vname, pname, _sargs, sign) in enumerate(intra):
            flat = f'_fi{i}_{suffix}'
            vi = self.t.vars[vname]
            idx_cols = set(self.r.all_names_for_var(vi))
            self._emit(f"{flat} = df_{vname}['{vname}'].reset_index()")
            if pname:
                pi = self.t.params[pname]
                param_idx_cols = []
                for s in pi.index_sets:
                    param_idx_cols.extend(self.r.names_for(s))
                on_repr = (f"'{param_idx_cols[0]}'" if len(param_idx_cols) == 1
                           else '[' + ', '.join(f"'{c}'" for c in param_idx_cols) + ']')
                self._emit(f"{flat} = pd.merge({flat}, s_{pi.pyomo_name.lower()}.reset_index(), on={on_repr})")
                idx_cols |= set(param_idx_cols)
            flat_infos.append({'df': flat, 'var': vname, 'param': pname,
                               'idx': idx_cols, 'sign': sign})

        # --- Step 2: iteratively merge all flat DataFrames on shared index columns ---
        merged = f'_mi_{suffix}'
        self._emit(f"{merged} = {flat_infos[0]['df']}")
        accumulated = flat_infos[0]['idx'].copy()
        for info in flat_infos[1:]:
            common = sorted(accumulated & info['idx'])
            on_repr = (f"'{common[0]}'" if len(common) == 1
                       else '[' + ', '.join(f"'{c}'" for c in common) + ']')
            self._emit(f"{merged} = pd.merge({merged}, {info['df']}, on={on_repr})")
            accumulated |= info['idx']

        # --- Step 3: build the linear combination expression string ---
        expr_str = ''
        for i, info in enumerate(flat_infos):
            vname, pname, sign = info['var'], info['param'], info['sign']
            if pname:
                pi = self.t.params[pname]
                term_str = f"{merged}['{pi.pyomo_name.lower()}'] * {merged}['{vname}']"
            else:
                term_str = f"{merged}['{vname}']"
            if i == 0:
                expr_str = f"-{term_str}" if sign < 0 else term_str
            else:
                expr_str += f" - {term_str}" if sign < 0 else f" + {term_str}"

        # --- Step 4: groupby outer constraint index and sum ---
        # After merge the table has a plain RangeIndex, so pass column references
        # explicitly (not string names) to groupby.
        lhs_var = f'lhs_{suffix}'
        grp_parts = [f"{merged}['{k}']" for k in outer_names]
        grp_repr = grp_parts[0] if len(grp_parts) == 1 else '[' + ', '.join(grp_parts) + ']'
        self._emit(f"{lhs_var} = ({expr_str}).groupby({grp_repr}).sum()")

        rhs_repr = self._rhs_repr(ci)
        sense = _gurobi_sense(ci.op)
        self._emit(f"gppd.add_constrs(m, {lhs_var}, {sense}, {rhs_repr}, name='{ci.pyomo_name}')")

    # ------------------------------------------------------------------
    def _gen_P6(self, ci: ConstrInfo):
        """Direct var access — always restricted to the constraint's declared index.

        Handles all three sub-cases uniformly:
          • constraint index == variable domain  (most common — .loc is identity)
          • constraint index is a proper subset  (whether or not declared with within=)
          • constraint index is a sparse dimen=2 set  (from_tuples, not from_product)
        """
        var_name = ci.lhs_direct_var
        df_var = f'df_{var_name}'
        suffix = ci.pyomo_name

        constr_idx = f'_idx_{suffix}'
        self._emit_constr_index(ci, constr_idx)
        lhs_repr = f"{df_var}['{var_name}'].loc[{constr_idx}]"

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


def solve(func, data, silent: bool = True):
    """
    Translate func, build and optimize the gurobipy-pandas model, and return
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
    import gurobipy as gp   # noqa: F401 — needed in exec'd namespace
    import gurobipy_pandas as gppd  # noqa: F401
    import pandas as pd     # noqa: F401

    raw_src = inspect.getsource(func)
    src = textwrap.dedent(raw_src)
    tree = ast.parse(src)
    func_def = tree.body[0]

    t = _Translator(func_def)
    t.parse()
    t.classify()
    registry = _IndexRegistry(t)
    registry.build()
    code = _CodeGen(t, registry).generate(extended=True)

    ns = {}
    exec(compile(code, '<translated-extended>', 'exec'), ns)
    model, var_series = ns['build_vectorized_model'](data)

    if silent:
        model.setParam('OutputFlag', 0)
    model.optimize()

    if model.SolCount == 0:
        return model, {}

    values = {name: series.apply(lambda v: v.X) for name, series in var_series.items()}
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
