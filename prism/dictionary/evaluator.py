"""Deterministic evaluation of an approved formula.

This is a security boundary as much as a correctness one. A formula originates
as model-authored text, so anything accepted here is something a model can make
PRISM execute - hence a restricted AST walker rather than eval().

It is also the reason the dictionary tier exists at all. Left to compute freely
with correct inputs in hand, both gpt-4o-mini and gpt-4o produced
defensible-but-unintended conventions (0.85 and 0.90 against an expected 0.96).
Once a convention is approved, the arithmetic happens here and only here.
"""
import ast
import operator

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv}


class FormulaError(ValueError):
    pass


def formula_facts(formula: str) -> list:
    """The identifiers a formula depends on - used to populate an entry's
    required_facts at approval time."""
    return sorted({n.id for n in ast.walk(ast.parse(formula, mode="eval"))
                   if isinstance(n, ast.Name)})


def evaluate_formula(formula: str, facts: dict) -> float:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"unparseable formula {formula!r}: {e}") from e

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise FormulaError(f"non-numeric constant {node.value!r}")
        if isinstance(node, ast.Name):
            # Never default a missing fact to zero: that yields a plausible,
            # confidently wrong number, which is the failure this tier prevents.
            if node.id not in facts:
                raise FormulaError(f"formula references unbound fact {node.id!r}")
            if facts[node.id] is None:
                raise FormulaError(f"fact {node.id!r} has no bound value")
            return float(facts[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise FormulaError(f"disallowed expression element: {type(node).__name__}")

    try:
        return walk(tree)
    except ZeroDivisionError as e:
        raise FormulaError("division by zero") from e
