from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    SYMPY_AVAILABLE = True
except Exception:
    # SymPy is optional at runtime. The checker still runs without symbolic matching.
    SYMPY_AVAILABLE = False
    N = None
    simplify = None
    parse_latex = None
    parse_expr = None
    standard_transformations = None
    implicit_multiplication_application = None


REASON_LITERAL_MATCH = "literal_match"
REASON_NUMERIC_MATCH = "numeric_match"
REASON_SYMBOLIC_MATCH = "symbolic_match"
REASON_NO_MATCH = "no_match"
REASON_CAN_NOT_EXTRACT = "can_not_extract"

TRAILING_TEXT_RE = re.compile(r"(.*?)(?:\\,|,)?\s*\\text\{([^{}]*)\}\s*$")
DEGREE_RE = re.compile(r"(?:\^\s*\{?\s*\\circ\s*\}?|\u00B0|\u00C2\u00B0)\s*$")
LATEX_TIMES_RE = re.compile(
    r"^([-+]?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?([-+]?\d+)\}?$"
)
COMPACT_UNIT_RE = re.compile(
    r"^(.*\d)\s*(cm|mm|km|m/s(?:\^2)?|m|kg|mg|g|ml|mL|L|s)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BoxedAnswer:
    content: str
    found: bool
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    reason: str
    extracted_answer: str


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _strip_math_wrappers(text: str) -> str:
    """Remove common wrapper delimiters without changing core content."""
    s = text.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    if len(s) >= 4 and s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if len(s) >= 4 and s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    if len(s) >= 6 and s.startswith(r"\\(") and s.endswith(r"\\)"):
        s = s[3:-3].strip()
    if len(s) >= 6 and s.startswith(r"\\[") and s.endswith(r"\\]"):
        s = s[3:-3].strip()
    return s


def _replace_text_commands(text: str) -> str:
    """Replace simple LaTeX text commands with their contents."""
    s = text
    for command in ("text", "mathrm", "operatorname", "mathbf"):
        pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")
        previous = None
        while previous != s:
            previous = s
            s = pattern.sub(r"\1", s)
    return s


def _normalize_latex_surface(text: str) -> str:
    """Normalize LaTeX presentation details that do not change the answer."""
    s = _strip_math_wrappers(to_text(text))
    s = s.replace("\u00a0", " ")
    s = s.replace("π", r"\pi")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\%", "%")
    for command in (r"\,", r"\!", r"\;", r"\:", r"\ "):
        s = s.replace(command, "")
    s = re.sub(r"\\begin\{[pbvBV]?matrix\}", r"\\begin{matrix}", s)
    s = re.sub(r"\\end\{[pbvBV]?matrix\}", r"\\end{matrix}", s)
    simple_script = r"([A-Za-z]|[-+]?\d+(?:\.\d+)?)"
    s = re.sub(rf"\^\{{\s*{simple_script}\s*\}}", r"^\1", s)
    s = re.sub(rf"_\{{\s*{simple_script}\s*\}}", r"_\1", s)
    s = re.sub(r"(\\tan|\\sin|\\cos)\{\s*([^{}]+?)\s*\}", r"\1 \2", s)
    s = _replace_text_commands(s)
    s = s.replace("~", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = _strip_redundant_outer_parentheses(s)
    return s


def _has_top_level_comma(text: str) -> bool:
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    for char in text:
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "," and not (brace_depth or paren_depth or bracket_depth):
            return True
    return False


def _strip_redundant_outer_parentheses(text: str) -> str:
    s = text.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s

    depth = 0
    for idx, char in enumerate(s):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and idx != len(s) - 1:
                return s

    inner = s[1:-1].strip()
    if not inner or _has_top_level_comma(inner):
        return s
    return inner


def _normalize_for_literal_compare(text: str) -> str:
    """Normalize formatting noise for robust literal comparison."""
    s = _normalize_latex_surface(to_text(text))
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def _is_unit_text(text: str) -> bool:
    """Return true for compact measurement-unit text, not answer words."""
    s = to_text(text).strip()
    if not s or s.startswith("-"):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9/*^\- °]*", s))


def _looks_like_numeric_quantity_text(text: str) -> bool:
    """Cheap guard used before stripping units from a larger answer string."""
    s = _normalize_latex_surface(to_text(text)).strip()
    if not s:
        return False
    if re.search(r"[A-Za-z]", s) and not re.search(r"\\(?:frac|sqrt|times|cdot|pi)\b", s):
        return False
    return bool(re.search(r"\d", s))


def _numbers_close(left: float, right: float) -> bool:
    """Strict numeric comparison that never rounds a nonzero value to zero."""
    if left == right:
        return True
    if left == 0 or right == 0:
        return abs(left - right) <= 1e-20
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)


def _is_safe_symbolic_text(text: str) -> bool:
    """Reject natural-language answer text before it reaches SymPy."""
    s = _normalize_latex_surface(text)
    allowed_words = {"frac", "sqrt", "sin", "cos", "tan", "log", "ln", "pi"}
    words = re.findall(r"\\?([A-Za-z]{2,})", s)
    return all(word in allowed_words or len(word) <= 2 for word in words)


def _literal_match(gt: str, pred: str) -> bool:
    gt_raw = to_text(gt).strip()
    pred_raw = to_text(pred).strip()
    if gt_raw == pred_raw:
        return True

    if _normalize_for_literal_compare(gt_raw) == _normalize_for_literal_compare(pred_raw):
        return True

    return False


def _parse_numeric_candidate(text: str) -> Optional[float]:
    """
    Parse a single numeric candidate from text.
    Supports int/float, percent, a/b fractions, and simple LaTeX fractions.
    """
    s = _strip_unit_suffix(to_text(text)).strip()
    if not s:
        return None

    s = s.replace(",", "")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    product = _parse_numeric_product(s)
    if product is not None:
        return product

    # Handle simple assignment forms like "x = 3".
    if s.count("=") == 1:
        left, right = s.split("=")
        if len(left.strip()) <= 2:
            s = right.strip()

    if s.endswith("%"):
        base = _parse_numeric_candidate(s[:-1])
        if base is None:
            return None
        return base

    frac_latex = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac_latex:
        numerator = _parse_numeric_candidate(frac_latex.group(1))
        denominator = _parse_numeric_candidate(frac_latex.group(2))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    frac_plain = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", s)
    if frac_plain:
        denominator = float(frac_plain.group(2))
        if denominator == 0:
            return None
        return float(frac_plain.group(1)) / denominator

    latex_times = LATEX_TIMES_RE.fullmatch(s)
    if latex_times:
        try:
            return float(latex_times.group(1)) * (10 ** int(latex_times.group(2)))
        except OverflowError:
            return None

    try:
        return float(s)
    except Exception:
        return None


def _parse_numeric_product(text: str) -> Optional[float]:
    s = _normalize_latex_surface(to_text(text)).strip()
    if not re.search(r"(?:\\times|\\cdot|\*)", s):
        return None

    pieces = re.split(r"\s*(?:\\times|\\cdot|\*)\s*", s)
    if len(pieces) < 2:
        return None

    value = 1.0
    for piece in pieces:
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", piece.strip()):
            return None
        value *= float(piece)
    return value


def _numeric_ground_truth_forms(text: str) -> set[float]:
    """Return accepted numeric forms for a ground-truth value."""
    s = _strip_unit_suffix(to_text(text)).strip()
    value = _parse_numeric_candidate(s)
    if value is None:
        return set()

    return {value, value / 100.0, value * 100.0}


def _strip_unit_suffix(text: str) -> str:
    """Drop trailing units, percent escapes, and degree markers without touching math."""
    s = _strip_math_wrappers(to_text(text)).strip()
    previous = None
    while previous != s:
        previous = s
        text_unit = TRAILING_TEXT_RE.fullmatch(s)
        if text_unit:
            prefix = text_unit.group(1).strip()
            unit = text_unit.group(2).strip()
            if _is_unit_text(unit) and _looks_like_numeric_quantity_text(prefix):
                s = prefix
        s = DEGREE_RE.sub("", s).strip()
    s = _normalize_latex_surface(s).strip()
    previous = None
    while previous != s:
        previous = s
        s = DEGREE_RE.sub("", s).strip()
        temperature_unit = re.fullmatch(
            r"(.*\d)\s*(?:\^\{?\\circ\}?|\u00B0|\u00C2\u00B0)\s*[CF]",
            s,
        )
        if temperature_unit:
            s = temperature_unit.group(1).strip()
            continue
        currency_value = re.fullmatch(
            r"(?:\\\$|\$|£|€)\s*([-+]?\d+(?:\.\d+)?)",
            s,
        )
        if currency_value:
            s = currency_value.group(1)
            continue
        inline_math_unit = re.fullmatch(
            r"(\$[^$]+\$)\s+[A-Za-z][A-Za-z0-9/*^\- ]*(?:/[A-Za-z][A-Za-z0-9/*^\- ]*)?",
            s,
        )
        if inline_math_unit:
            s = _strip_math_wrappers(inline_math_unit.group(1)).strip()
            continue
        # Handle normalized forms such as "3.34 kJ" and "-21.7 J/mol K".
        bare_unit = re.fullmatch(
            r"(.*\d)\s+[A-Za-z][A-Za-z0-9/*^\- ]*(?:/[A-Za-z][A-Za-z0-9/*^\- ]*)?",
            s,
        )
        if bare_unit and _looks_like_numeric_quantity_text(bare_unit.group(1)):
            s = bare_unit.group(1).strip()
            continue
        compact_unit = COMPACT_UNIT_RE.fullmatch(s)
        if compact_unit and _looks_like_numeric_quantity_text(compact_unit.group(1)):
            s = compact_unit.group(1).strip()
    return s.rstrip(",;；，").strip()


def _numeric_match(gt: str, pred: str) -> bool:
    """
    Compare numeric values with lenient percent/unit-scale tolerance.

    A prediction matches a ground-truth number when it equals g, g / 100, or
    g * 100. This intentionally allows common percent/unit-scale slips.
    """
    gt_forms = _numeric_ground_truth_forms(gt)
    pred_value = _parse_numeric_candidate(pred)
    if not gt_forms or pred_value is None:
        return False

    for g in gt_forms:
        if _numbers_close(pred_value, g):
            return True
    return False


def _parse_symbolic_expr(text: str):
    if not SYMPY_AVAILABLE:
        return None

    s = _strip_unit_suffix(to_text(text)).strip()
    if not s:
        return None
    if not _is_safe_symbolic_text(s):
        return None

    # Try LaTeX parser first, then a lightweight LaTeX-to-SymPy text fallback.
    for parser in (parse_latex, parse_expr):
        try:
            if parser is parse_expr:
                transformations = standard_transformations + (
                    implicit_multiplication_application,
                )
                return parser(_latex_to_sympy_text(s), transformations=transformations)
            return parser(s)
        except Exception:
            continue
    return None


def _symbolic_exprs_equal(left, right) -> bool:
    try:
        if left == right or str(left) == str(right):
            return True
    except Exception:
        pass

    try:
        if simplify(left - right) == 0:
            return True
    except Exception:
        pass

    try:
        if left.equals(right):
            return True
    except Exception:
        pass

    return False


def _symbolic_exprs_proportional(left, right) -> bool:
    if _symbolic_exprs_equal(left, right):
        return True

    try:
        if simplify(left) == 0 or simplify(right) == 0:
            return False
    except Exception:
        return False

    try:
        ratio = simplify(left / right)
        if ratio == 0:
            return False
        if not getattr(ratio, "free_symbols", set()):
            return True
    except Exception:
        pass

    return False


def _parse_simple_polynomial(text: str) -> dict[tuple[str, ...], float] | None:
    s = _normalize_latex_surface(text).replace(" ", "")
    if not s or not _is_safe_symbolic_text(s):
        return None
    if any(ch in s for ch in "^=/\\"):
        return None

    factors = re.fullmatch(r"\(([^()]+)\)\(([^()]+)\)", s)
    if factors:
        left = _parse_simple_polynomial(factors.group(1))
        right = _parse_simple_polynomial(factors.group(2))
        if left is None or right is None:
            return None
        product: dict[tuple[str, ...], float] = {}
        for left_vars, left_coeff in left.items():
            for right_vars, right_coeff in right.items():
                vars_key = tuple(sorted(left_vars + right_vars))
                product[vars_key] = product.get(vars_key, 0.0) + left_coeff * right_coeff
        return {key: value for key, value in product.items() if value}

    terms = re.findall(r"[+-]?[^+-]+", s)
    if not terms or "".join(terms) != s:
        return None

    polynomial: dict[tuple[str, ...], float] = {}
    for term in terms:
        match = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)?([A-Za-z]*)", term)
        if not match:
            return None
        sign = -1.0 if match.group(1) == "-" else 1.0
        coeff = float(match.group(2)) if match.group(2) else 1.0
        vars_key = tuple(sorted(match.group(3)))
        polynomial[vars_key] = polynomial.get(vars_key, 0.0) + sign * coeff

    return {key: value for key, value in polynomial.items() if value}


def _fallback_symbolic_match(gt: str, pred: str) -> bool:
    gt_poly = _parse_simple_polynomial(gt)
    pred_poly = _parse_simple_polynomial(pred)
    return gt_poly is not None and pred_poly is not None and gt_poly == pred_poly


def _consume_braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None

    depth = 1
    idx = start + 1
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
        idx += 1
    return None


def _replace_latex_command_two_args(text: str, command: str, template: str) -> str:
    marker = f"\\{command}"
    s = text
    while marker in s:
        start = s.find(marker)
        first = _consume_braced(s, start + len(marker))
        if first is None:
            break
        second = _consume_braced(s, first[1])
        if second is None:
            break
        replacement = template.format(
            _latex_to_sympy_text(first[0]),
            _latex_to_sympy_text(second[0]),
        )
        s = s[:start] + replacement + s[second[1] :]
    return s


def _replace_latex_command_one_arg(text: str, command: str, template: str) -> str:
    marker = f"\\{command}"
    s = text
    while marker in s:
        start = s.find(marker)
        arg = _consume_braced(s, start + len(marker))
        if arg is None:
            break
        replacement = template.format(_latex_to_sympy_text(arg[0]))
        s = s[:start] + replacement + s[arg[1] :]
    return s


def _latex_to_sympy_text(text: str) -> str:
    """Convert common answer-level LaTeX into parse_expr-compatible text."""
    s = _normalize_latex_surface(text)
    s = _replace_latex_command_two_args(s, "frac", "(({0})/({1}))")
    s = _replace_latex_command_one_arg(s, "sqrt", "sqrt({0})")
    s = s.replace(r"\pi", "pi")
    for command in (r"\cdot", r"\times"):
        s = s.replace(command, "*")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\(?:sin|cos|tan|log|ln)\s*", lambda m: m.group(0)[1:], s)
    return s


def _symbolic_match(gt: str, pred: str) -> bool:
    gt_text = _strip_unit_suffix(to_text(gt)).strip()
    pred_text = _strip_unit_suffix(to_text(pred)).strip()
    if _fallback_symbolic_match(gt_text, pred_text):
        return True

    if not SYMPY_AVAILABLE:
        return False

    # Equation-to-equation comparison: compare normalized residual forms.
    both_equations = gt_text.count("=") == 1 and pred_text.count("=") == 1
    if both_equations:
        gt_l, gt_r = gt_text.split("=")
        pr_l, pr_r = pred_text.split("=")
        gt_text = f"({gt_l})-({gt_r})"
        pred_text = f"({pr_l})-({pr_r})"

    gt_expr = _parse_symbolic_expr(gt_text)
    pred_expr = _parse_symbolic_expr(pred_text)
    if gt_expr is None or pred_expr is None:
        return False

    if both_equations:
        return _symbolic_exprs_proportional(gt_expr, pred_expr)

    if _symbolic_exprs_equal(gt_expr, pred_expr):
        return True

    try:
        gt_num = float(N(gt_expr))
        pred_num = float(N(pred_expr))
        if _numbers_close(gt_num, pred_num):
            return True
    except Exception:
        pass

    return False


def _split_top_level_equals(text: str) -> list[str]:
    s = to_text(text)
    pieces: list[str] = []
    start = 0
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0

    for idx, char in enumerate(s):
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "=" and not (brace_depth or paren_depth or bracket_depth):
            pieces.append(s[start:idx].strip())
            start = idx + 1

    pieces.append(s[start:].strip())
    return [piece for piece in pieces if piece]


def _looks_like_assignment_lhs(text: str) -> bool:
    s = _normalize_latex_surface(text).replace(" ", "")
    return bool(
        re.fullmatch(
            r"(?:[A-Za-z]|[A-Za-z](?:\^-?\d+)?\([A-Za-z]\)|[A-Za-z]_\w+|[A-Za-z]\^-?\d+)",
            s,
        )
    )


def _is_zero_text(text: str) -> bool:
    s = _normalize_for_literal_compare(text)
    return s in {"0", "+0", "-0", "0.0", "+0.0", "-0.0"}


def _identity_sides_equivalent(left: str, right: str) -> bool:
    if _is_zero_text(left) or _is_zero_text(right):
        return False

    if SYMPY_AVAILABLE:
        left_expr = _parse_symbolic_expr(left)
        right_expr = _parse_symbolic_expr(right)
        if left_expr is not None and right_expr is not None:
            return _symbolic_exprs_equal(left_expr, right_expr)

    # Fallback for common expanded/factored identity answers when SymPy is absent.
    left_norm = _normalize_latex_surface(left)
    right_norm = _normalize_latex_surface(right)
    left_has_factor = re.search(r"\([^()]*[A-Za-z][^()]*\)", left_norm)
    right_has_factor = re.search(r"\([^()]*[A-Za-z][^()]*\)", right_norm)
    return bool(
        (left_has_factor or right_has_factor)
        and re.search(r"[A-Za-z]", left_norm)
        and re.search(r"[A-Za-z]", right_norm)
        and not _is_zero_text(left_norm)
        and not _is_zero_text(right_norm)
    )


def _answer_candidates(text: str) -> list[str]:
    """Return safe whole-answer candidates; never substring-match arbitrary text."""
    raw = to_text(text).strip()
    normalized = _normalize_latex_surface(raw)
    stripped_units = _strip_unit_suffix(normalized)

    candidates: list[str] = []
    for value in (raw, normalized, stripped_units):
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    boxed = find_last_boxed_answer(raw)
    if boxed.found:
        for value in (boxed.content, _normalize_latex_surface(boxed.content), _strip_unit_suffix(boxed.content)):
            value = value.strip()
            if value and value not in candidates:
                candidates.append(value)

    parts = _split_top_level_equals(normalized)
    if len(parts) == 2:
        left, right = parts
        if _looks_like_assignment_lhs(left):
            for value in (right, _strip_unit_suffix(right)):
                value = value.strip()
                if value and value not in candidates:
                    candidates.append(value)
        elif _identity_sides_equivalent(left, right):
            for value in parts:
                stripped = _strip_unit_suffix(value)
                for candidate in (value, stripped):
                    candidate = candidate.strip()
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)

    membership = re.fullmatch(r"[A-Za-z]\s*\\in\s*(.+)", normalized)
    if membership:
        value = membership.group(1).strip()
        if value and value not in candidates:
            candidates.append(value)

    set_literal = re.fullmatch(r"\\\{\s*(.+?)\s*\\\}", normalized)
    if set_literal:
        value = set_literal.group(1).strip()
        if value and not _has_top_level_comma(value) and value not in candidates:
            candidates.append(value)

    return candidates


def _valid_boxed_answers(text: Any) -> list[BoxedAnswer]:
    full_text = to_text(text)
    marker = r"\boxed{"
    marker_len = len(marker)
    answers: list[BoxedAnswer] = []

    for match in re.finditer(re.escape(marker), full_text):
        content_start = match.start() + marker_len
        depth = 1
        idx = content_start
        while idx < len(full_text):
            char = full_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(
                        BoxedAnswer(
                            content=full_text[content_start:idx].strip(),
                            found=True,
                            start=match.start(),
                            end=idx + 1,
                        )
                    )
                    break
            idx += 1

    return answers


def find_last_boxed_answer(text: Any) -> BoxedAnswer:
    """
    Find the last valid \\boxed{...} and keep its source span.

    If none is valid, content is the full text and found is False.
    """
    full_text = to_text(text)
    answers = _valid_boxed_answers(full_text)
    if not answers:
        return BoxedAnswer(content=full_text.strip(), found=False)
    return answers[-1]


def extract_boxed_answer(text: Any) -> Tuple[str, bool]:
    """
    Extract content from the last valid \\boxed{...}.

    Returns:
    - (extracted_content, True) when a valid boxed answer is found.
    - (full_text, False) otherwise.
    """
    answer = find_last_boxed_answer(text)
    return answer.content, answer.found


def remove_valid_boxed_expressions(text: Any) -> str:
    """Remove all valid \\boxed{...} spans from text."""
    full_text = to_text(text)
    answers = _valid_boxed_answers(full_text)
    if not answers:
        return full_text

    pieces: list[str] = []
    last_end = 0
    for answer in answers:
        if answer.start is None or answer.end is None:
            continue
        pieces.append(full_text[last_end:answer.start])
        last_end = answer.end
    pieces.append(full_text[last_end:])
    return "".join(pieces)


def match_answer(
    gt: str | int | float,
    pred_text: str | int | float,
    question_text: Any = None,
) -> MatchResult:
    """
    Match a model prediction against ground truth.

    Rules:
    - If no valid boxed answer is found, the sample is incorrect with can_not_extract.
    - If boxed answer exists, reason reflects the engine that matched.
    """
    gt_text = to_text(gt)
    answer = find_last_boxed_answer(pred_text)
    extracted_answer = answer.content

    if not answer.found:
        return MatchResult(False, REASON_CAN_NOT_EXTRACT, extracted_answer)

    gt_candidates = _answer_candidates(gt_text)
    pred_candidates = _answer_candidates(extracted_answer)

    try:
        for gt_candidate in gt_candidates:
            for pred_candidate in pred_candidates:
                if _literal_match(gt_candidate, pred_candidate):
                    return MatchResult(True, REASON_LITERAL_MATCH, extracted_answer)

        for gt_candidate in gt_candidates:
            for pred_candidate in pred_candidates:
                if _numeric_match(gt_candidate, pred_candidate):
                    return MatchResult(True, REASON_NUMERIC_MATCH, extracted_answer)

        for gt_candidate in gt_candidates:
            for pred_candidate in pred_candidates:
                if _symbolic_match(gt_candidate, pred_candidate):
                    return MatchResult(True, REASON_SYMBOLIC_MATCH, extracted_answer)
    except Exception:
        # Keep output contract stable even when one row is malformed.
        return MatchResult(False, REASON_NO_MATCH, extracted_answer)

    return MatchResult(False, REASON_NO_MATCH, extracted_answer)
