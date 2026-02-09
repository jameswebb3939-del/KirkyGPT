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
    Validates a list of follow-up questions formatted as bullet points.
    
    Args:
        text: Raw text containing bullet items
        min_questions: Minimum number of items required
        bullet_style: Which bullet characters to accept
        require_question_mark: Whether each item must end with ?
        forbid_extra_text: Whether to disallow non-bullet lines
    
    Returns:
        ValidationResult with validation status, item count, errors, and normalized text
    """
    text = text.strip()
    errors: list[str] = []
    # Split into lines and filter empty ones
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        errors.append("No content provided")
        return ValidationResult(ok=False, num_items=0, errors=errors, normalized_text=None)
    # Extract bullet items
    bullet_items: list[str] = []
    line_number = 0
    for line_number, line in enumerate(lines, 1):
        # Check if line starts with a bullet
        is_dash = line.startswith('-')
        is_asterisk = line.startswith('*')
        if not (is_dash or is_asterisk):
            # Non-bullet line found
            if forbid_extra_text:
                errors.append(f"Found non-bullet text on line {line_number}: '{line}'")
            continue
        # Validate bullet style
        if bullet_style == "dash" and not is_dash:
            errors.append(f"Line {line_number} uses * but style is 'dash' only")
            continue
        elif bullet_style == "asterisk" and not is_asterisk:
            errors.append(f"Line {line_number} uses - but style is 'asterisk' only")
            continue
        # Extract content (remove bullet and whitespace)
        content = line[1:].strip()
        if not content:
            errors.append(f"Line {line_number} has empty content after bullet")
            continue
        # Check for question mark if required
        if require_question_mark and not content.endswith('?'):
            errors.append(f"Item {len(bullet_items) + 1} does not end with ?")
        bullet_items.append(content)
    num_items = len(bullet_items)
    # Check minimum questions requirement
    if num_items < min_questions:
        errors.append(f"Only {num_items} questions found, need at least {min_questions}")
    # Determine if validation passed
    ok = len(errors) == 0    
    # Build normalized text
    normalized_text = None
    if ok:
        if bullet_style == "asterisk":
            normalized_text = '\n'.join(f"* {item}" for item in bullet_items)
        else:  # dash or either, default to dash
            normalized_text = '\n'.join(f"- {item}" for item in bullet_items)
    return ValidationResult(ok=ok, num_items=num_items, errors=errors, normalized_text=normalized_text)

def normalize_bullets(text: str, *, style: Literal["dash","asterisk"] = "dash") -> str:
    """
    Normalize bullets in `text`:

    - preserve line order but only keep bullet lines (starting with '-' or '*')
    - strip the bullet marker and normalize internal whitespace
    - skip empty or non-bullet lines
    - emit bullets using the chosen `style` ("dash" => '-', "asterisk" => '*')
    - one space after the bullet, one bullet per line, joined by '\n'
    """
    out_lines: list[str] = []
    bullet_char = "*" if style == "asterisk" else "-"

    for line in text.splitlines():
        if not line:
            continue
        # match a bullet at start (allow leading whitespace)
        m = re.match(r"^\s*([-\*])\s*(.*)$", line)
        if not m:
            # not a bullet line -> skip
            continue
        content = m.group(2).strip()
        if not content:
            # empty content after bullet -> skip
            continue
        # normalize internal whitespace
        content = " ".join(content.split())
        out_lines.append(f"{bullet_char} {content}")

    return "\n".join(out_lines)


def extract_bullet_items(text: str, *, bullet_style: Literal["dash","asterisk","either"] = "either") -> list[str]:
    """
    Extract bullet item contents from text.
    
    Args:
        text: Text containing bullet items
        bullet_style: "dash" (only -), "asterisk" (only *), or "either" (- or *)
    
    Returns:
        list[str] of bullet contents (without markers), in original order
    """
    # Step 1: Normalize input
    if text is None or not text or not text.strip():
        return []
    text = text.strip()
    
    # Step 3: Determine valid bullet prefixes
    valid_prefixes = set()
    if bullet_style in ("dash", "either"):
        valid_prefixes.add("-")
    if bullet_style in ("asterisk", "either"):
        valid_prefixes.add("*")
    
    # Step 2 & 4-8: Split lines and extract bullets
    bullets = []
    for line in text.splitlines():
        # Step 4: Strip whitespace
        line = line.strip()
        if not line:
            # Step 4: Skip empty lines
            continue
        
        # Step 5: Detect bullet (first char must be a valid prefix)
        if line[0] not in valid_prefixes:
            # Not a valid bullet line -> skip
            continue
        
        # Step 6: Extract content (remove prefix and optional space after)
        content = line[1:]
        if content.startswith(" "):
            content = content[1:]
        content = content.strip()
        
        # Step 7: Store if non-empty
        if content:
            bullets.append(content)
    
    return bullets


def try_repair_to_followups(text: str, *, min_questions: int = 3, bullet_style: Literal["dash","asterisk"] = "dash") -> str | None:
    """
    Try to repair a piece of text into a normalized bullet list.

    Follows the pseudocode:
    - return None for empty input
    - extract bullets allowing either marker
    - keep only items that end with '?'
    - require at least `min_questions`
    - emit bullets using the requested `bullet_style`
    """
    # Step 1: Guard clauses
    if text is None or not text.strip():
        return None

    # Step 2: Extract bullets (allow either style when repairing)
    items = extract_bullet_items(text, bullet_style="either")

    # Step 3: Filter invalid items (strip and require trailing '?')
    valid_items: list[str] = []
    for item in items:
        s = item.strip()
        if not s:
            continue
        if not s.endswith("?"):
            continue
        valid_items.append(s)

    # Step 4: Check minimum viability
    if len(valid_items) < min_questions:
        return None

    # Step 5: Ensure items are clean (no leading bullets, trimmed)
    normalized_items = [it.strip().lstrip("-* ") for it in valid_items]

    # Step 6: Rebuild normalized bullet list
    bullet_prefix = "*" if bullet_style == "asterisk" else "-"
    lines = [f"{bullet_prefix} {it}" for it in normalized_items]
    normalized_text = "\n".join(lines)

    # Step 7: Final cleanup
    normalized_text = normalized_text.strip()

    # Step 8: Return repaired output
    return normalized_text



def fallback_followups(prompt_summary: str | None = None, *, min_questions: int = 3, bullet_style: Literal["dash","asterisk"] = "dash") -> str:
    """
    Generate a guaranteed valid bullet list of follow-up questions.

    Always returns at least `min_questions` bullet items, each ending with '?'.
    Optionally uses `prompt_summary` to tailor wording slightly (topic hint).
    
    Returns a string with one bullet per line, never prose outside bullets.
    """
    # Step 1: Choose bullet prefix
    prefix = "*" if bullet_style == "asterisk" else "-"

    # Step 2: Build optional topic hint
    topic_hint: str | None = None
    if prompt_summary:
        prompt_summary = prompt_summary.strip()
        if prompt_summary:
            # Extract first ~8-12 words as hint
            words = prompt_summary.split()[:10]
            topic_hint = " ".join(words)

    # Step 3: Prepare template libraries
    templates = [
        "What outcome do you want?",
        "What constraints or requirements should I follow?",
        "What inputs or examples can you share?",
        "What edge cases should be handled?",
        "What should the response format look like?",
        "What should I do first?",
    ]

    topic_templates = []
    if topic_hint:
        topic_templates = [
            f"What is the main goal for {topic_hint}?",
            f"What constraints apply to {topic_hint}?",
            f"What examples can you provide for {topic_hint}?",
        ]

    # Step 4: Select questions to meet min_questions
    questions: list[str] = []
    seen: set[str] = set()

    # Add topic-specific templates first (if available)
    if topic_templates:
        for q in topic_templates[:2]:
            if q not in seen:
                questions.append(q)
                seen.add(q)

    # Fill remaining from generic templates
    template_idx = 0
    while len(questions) < min_questions:
        q = templates[template_idx % len(templates)]
        if q not in seen:
            questions.append(q)
            seen.add(q)
        template_idx += 1

    # Step 5: Enforce strict validity on each question
    clean_questions: list[str] = []
    for q in questions:
        q = q.strip()
        # Replace internal newlines with spaces
        q = q.replace("\n", " ")
        # Ensure ends with ?
        if not q.endswith("?"):
            q += "?"
        clean_questions.append(q)

    # Step 6: Emit bullet list
    lines = [f"{prefix} {q}" for q in clean_questions]
    output_text = "\n".join(lines)

    # Step 7: Final checks and return
    output_text = output_text.strip()
    return output_text
        
            