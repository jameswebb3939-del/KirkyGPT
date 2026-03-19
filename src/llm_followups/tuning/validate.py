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


def try_repair_to_followups(text: str, *, min_questions: int = 3, bullet_style: Literal["dash","asterisk"] = "dash") -> str | None:
    """
    Attempt to repair malformed text into a valid bullet list of questions.

    Extracts bullets, filters for question-marked items, and rebuilds normalized list.
    
    Args:
        text: Text to repair.
        min_questions: Minimum questions required for valid output.
        bullet_style: Output bullet style ("dash" or "asterisk").
    
    Returns:
        Repaired bullet list, or None if insufficient valid questions found.
    """
    # Step 1: Guard clauses
    if text is None or not text.strip():
        return None

    # Step 2: Extract bullets (allow either style when repairing)
    items = extract_bullet_items(text, bullet_style="either")

    # Step 3: Filter invalid items (normalize and require trailing '?')
    valid_items: list[str] = []
    for item in items:
        s = normalize_item_text(item)
        if not s:
            continue
        if not s.endswith("?"):
            continue
        valid_items.append(s)

    # Step 4: Check minimum viability
    if len(valid_items) < min_questions:
        return None

    # Step 5: Rebuild normalized bullet list
    bullet_prefix = "*" if bullet_style == "asterisk" else "-"
    lines = [f"{bullet_prefix} {it}" for it in valid_items]
    normalized_text = "\n".join(lines)

    # Step 6: Return repaired output
    return normalized_text



def fallback_followups(prompt_summary: str | None = None, *, min_questions: int = 3, bullet_style: Literal["dash","asterisk"] = "dash") -> str:
    """
    Generate a guaranteed valid bullet list of follow-up questions.

    Always returns at least min_questions bullet items, each ending with '?'.
    Optionally uses prompt_summary to tailor wording (topic-specific generation).
    
    Args:
        prompt_summary: Optional hint text for generating topic-specific questions.
        min_questions: Minimum number of questions to generate.
        bullet_style: Output bullet style ("dash" or "asterisk").
    
    Returns:
        String with one bullet per line, guaranteed valid format.
    """
    prefix = "*" if bullet_style == "asterisk" else "-"

    templates = [
        "What specific goal are you trying to achieve?",
        "What constraints or requirements should I consider?",
        "What would a successful result look like?",
        "What tools, technologies, or resources are you already using?",
        "What is the biggest difficulty you are facing right now?",
        "Are there any examples or references you want me to follow?",
    ]

    topic_templates: list[str] = []
    if prompt_summary:
        topic_hint = " ".join(prompt_summary.strip().split()[:12])
        if topic_hint:
            topic_templates = [
                f"What part of '{topic_hint}' do you want to focus on first?",
                f"What constraints or requirements matter most for '{topic_hint}'?",
                f"What result are you hoping to achieve with '{topic_hint}'?",
                f"What tools, examples, or prior work do you already have for '{topic_hint}'?",
            ]

    questions: list[str] = []
    seen: set[str] = set()

    for q in topic_templates:
        if q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= min_questions:
            break

    template_idx = 0
    while len(questions) < min_questions:
        q = templates[template_idx % len(templates)]
        if q not in seen:
            questions.append(q)
            seen.add(q)
        template_idx += 1

    clean_questions = []
    for q in questions:
        q = q.strip().replace("\n", " ")
        q = q.replace("??", "?")
        if not q.endswith("?"):
            q += "?"
        clean_questions.append(q)

    return "\n".join(f"{prefix} {q}" for q in clean_questions)

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