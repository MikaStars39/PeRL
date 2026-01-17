import logging
import sys

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("Error: Please install math-verify. Run `pip install math-verify`")
    sys.exit(1)

logger = logging.getLogger("RM")


# ------------ Boxed Answer Extraction ------------

def extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...} or \\fbox{...} in text."""
    boxed_str = _find_last_boxed(text)
    if boxed_str is None:
        return None
    return _remove_boxed_wrapper(boxed_str)


def _find_last_boxed(text: str) -> str | None:
    """Find the last \\boxed{...} or \\fbox{...} substring."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    # Find matching closing brace
    brace_count = 0
    end_idx = None
    for i in range(idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    return text[idx : end_idx + 1] if end_idx else None


def _remove_boxed_wrapper(s: str) -> str:
    """Remove \\boxed{...} wrapper and return inner content."""
    # Handle "\\boxed " format (space after boxed)
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    # Handle "\\boxed{...}" format
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    # Handle "\\fbox{...}" format
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{"):-1]
    return s


# ------------ Score Computation ------------

def compute_score(
    prediction: str,
    ground_truth: str,
    timeout_score: float = 0.0,
) -> float:
    """Compute math equivalence score between prediction and ground truth."""
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    # Wrap in $ for latex parsing
    pred_fmt = f"${prediction}$"
    gold_fmt = f"${ground_truth}$"

    try:
        score, _ = verify_func([gold_fmt], [pred_fmt])
        return float(score)
    except TimeoutException:
        return timeout_score
    except Exception:
        return 0.0
