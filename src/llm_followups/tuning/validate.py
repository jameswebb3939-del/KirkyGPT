from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Literal

@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    num_items: int
    errors: list[str]
    normalized_text: str | None=None

def validate_followup_list(text: str, *, min_questions: int = 3, bullet_style: Literal["dash","asterisk","either"] = "either", require_question_mark: bool = True, forbid_extra_text: bool = True) -> ValidationResult:
    """
    Validate a list of follow-up questions formatted as bullet points.
    
    Checks for proper bullet markers, required question marks, and minimum count.
    
    Args:
        text: Text to validate as bullet list.
        min_questions: Minimum number of questions required.
        bullet_style: Accepted bullet markers ("dash", "asterisk", or "either").
        require_question_mark: Whether each item must end with '?'.
        forbid_extra_text: Whether non-bullet text is an error.
    
    Returns:
        ValidationResult with validation status and any errors.
    """
    if text is None or text.strip() == "":
        return ValidationResult(False, 0, ["No content provided"], None)

    errors: list[str] = []
    items: list[str] = []
    lines = text.splitlines()

    if bullet_style == "dash":
        accepted_markers = {"-"}
    elif bullet_style == "asterisk":
        accepted_markers = {"*"}
    else:
        accepted_markers = {"-", "*"}

    for idx, line in enumerate(lines, 1):
        if not line.strip():
            continue
        marker, content = parse_bullet_line(line)
        if marker is None:
            if forbid_extra_text:
                errors.append(f"Found non-bullet text on line {idx}: '{line.rstrip()}'.")
            continue
        if marker not in accepted_markers:
            errors.append(f"Wrong bullet marker '{marker}' on line {idx}: '{line.rstrip()}'.")
            continue
        content_norm = normalize_item_text(content)
        if not content_norm:
            errors.append(f"Empty content after bullet on line {idx}.")
            continue
        item_errs = validate_item(content_norm, require_question_mark=require_question_mark)
        for err in item_errs:
            errors.append(f"Line {idx}: {err}")
        if not item_errs:
            items.append(content_norm)

    if len(items) < min_questions:
        errors.append(f"Only {len(items)} questions found, need at least {min_questions}.")

    ok = (len(errors) == 0)
    normalized_text = None
    if ok:
        out_style = choose_output_bullet_style(bullet_style)
        marker = "-" if out_style == "dash" else "*"
        normalized_text = "\n".join(f"{marker} {item}" for item in items)

    return ValidationResult(ok, len(items), errors, normalized_text)

def normalize_bullets(text: str, *, style: Literal["dash","asterisk"] = "dash") -> str:
    """
    Normalize bullets in text to consistent format.

    Preserves line order, keeps only bullet lines, normalizes whitespace,
    and emits bullets using the chosen style.
    
    Args:
        text: Text with bullets to normalize.
        style: Output bullet style ("dash" or "asterisk").
    
    Returns:
        Text with normalized bullets.
    """
    out_lines: list[str] = []
    bullet_char = "*" if style == "asterisk" else "-"
    marker_set = accepted_markers_for_style("either")
    for line in text.splitlines():
        marker, content = parse_bullet_line(line)
        if marker not in marker_set:
            continue
        content_norm = normalize_item_text(content)
        if not content_norm:
            continue
        out_lines.append(f"{bullet_char} {content_norm}")
    return '\n'.join(out_lines)


def extract_bullet_items(text: str, *, bullet_style: Literal["dash","asterisk","either"] = "either") -> list[str]:
    """
    Extract bullet-pointed items from text.
    
    Args:
        text: Text to extract bullets from.
        bullet_style: Accepted bullet markers ("dash", "asterisk", or "either").
    
    Returns:
        List of extracted bullet item contents.
    """
    if text is None or not text.strip():
        return []
    accepted_markers = accepted_markers_for_style(bullet_style)
    items = []
    for line in text.splitlines():
        marker, content = parse_bullet_line(line)
        if marker in accepted_markers:
            content_norm = normalize_item_text(content)
            if content_norm:
                items.append(content_norm)
    return items



def try_repair_to_followups(text: str, *, min_questions: int = 3, bullet_style: Literal["dash", "asterisk"] = "dash") -> str | None:
    """
    Attempt to repair malformed text into a valid bullet list of questions.

    Accepts:
    - dash bullets
    - asterisk bullets
    - numbered lines like "1. ..."
    - mixed forms like "- 2. ..."

    Returns a normalized bullet list if at least min_questions valid questions are found.
    """
    if text is None or not text.strip():
        return None

    bullet_prefix = "*" if bullet_style == "asterisk" else "-"
    valid_items: list[str] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        s = raw_line.strip()
        if not s:
            continue

        # Strip leading bullet marker first if present
        s = re.sub(r"^[\-\*]\s*", "", s)

        # Strip leading numbering like "1. ", "2. ", etc.
        s = re.sub(r"^\d+\.\s*", "", s)

        s = normalize_item_text(s)
        if not s:
            continue

        if not s.endswith("?"):
            s = s.rstrip(".") + "?"

        if len(s) < 5:
            continue

        if s in seen:
            continue

        seen.add(s)
        valid_items.append(s)

    if len(valid_items) < min_questions:
        return None

    lines = [f"{bullet_prefix} {it}" for it in valid_items[:min_questions]]
    return "\n".join(lines)



def fallback_followups(prompt_summary: str | None = None, *, min_questions: int = 3, bullet_style: Literal["dash", "asterisk"] = "dash") -> str:
    """
    Generate a guaranteed valid bullet list of follow-up questions.

    Tries to stay topic-aware using prompt_summary instead of falling back to the
    same generic requirement questions every time.
    """
    prefix = "*" if bullet_style == "asterisk" else "-"
    topic = (prompt_summary or "this topic").strip()
    topic_lower = topic.lower()

    if "docker" in topic_lower or "container" in topic_lower or "compose" in topic_lower:
        questions = [
            "Are you trying to understand Docker conceptually, or use it in a real project?",
            "Do you want help with Dockerfiles, containers, or Docker Compose?",
            "Are you using Docker for local development, deployment, or both?",
        ]
    elif "pytest" in topic_lower or "test" in topic_lower:
        questions = [
            "Are you trying to learn pytest basics, or debug a failing test?",
            "Do you need help with assertions, fixtures, or test structure?",
            "Are you working with unit tests, integration tests, or async tests?",
        ]
    elif "fastapi" in topic_lower or "api" in topic_lower:
        questions = [
            "Are you building a new API or modifying an existing FastAPI service?",
            "Do you need help with routes, request validation, or response models?",
            "Are you working with async endpoints, dependency injection, or testing?",
        ]
    elif "sqlalchemy" in topic_lower or "database" in topic_lower:
        questions = [
            "Are you using SQLAlchemy with synchronous or asynchronous sessions?",
            "Do you need help with models, queries, or session management?",
            "Is your main issue related to setup, transactions, or integration into your app?",
        ]
    else:
        questions = [
            f"What specific part of {topic} do you want help with most?",
            f"Are you looking for a conceptual explanation of {topic}, or practical steps?",
            f"What are you trying to achieve with {topic} right now?",
        ]

    questions = questions[: max(min_questions, 3)]

    clean_questions: list[str] = []
    for q in questions:
        s = q.strip().replace("\n", " ")
        if not s.endswith("?"):
            s += "?"
        clean_questions.append(s)

    return "\n".join(f"{prefix} {q}" for q in clean_questions[:min_questions])



def accepted_markers_for_style(bullet_style: Literal["dash","asterisk","either"]) -> set[str]:
    if bullet_style == "dash":
        return {"-"}
    elif bullet_style == "asterisk":
        return {"*"}
    else:
        return {"-", "*"}

def choose_output_bullet_style(bullet_style: Literal["dash", "asterisk", "either"], *, default: Literal["dash","asterisk"]="dash") -> Literal["dash", "asterisk"]:
    if bullet_style == "dash" or bullet_style == "asterisk":
        return bullet_style
    return default

def parse_bullet_line(line: str) -> tuple[str | None, str]:
    raw = line.rstrip("\n")
    s = raw.lstrip()
    if s == "":
        return None, ""
    if s[0] == "-" or s[0] == "*":
        marker = s[0]
        content = s[1:].strip()
        return marker, content
    else:
        return None, raw.strip()
    
def normalize_item_text(item: str):
    s = item.strip()
    s = " ".join(s.split())
    s = s.replace("??", "?")
    return s

def validate_item(item: str, *, require_question_mark: bool = True) -> list[str]:
    errs = []
    s = item.strip()
    if s == "":
        errs.append("Empty bullet item")
    if require_question_mark and not s.endswith("?"):
        errs.append("Bullet does not end with ?")
    if s == "?" or len(s) < 5:
        errs.append("Too short")
    return errs