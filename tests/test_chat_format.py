from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from pydantic import ValidationError

from src.llm_followups.server.schemas import ChatRequest
from src.llm_followups.tuning.validate import validate_followup_list


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for JSONL validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate SFT JSONL chat format."
    )
    
    parser.add_argument(
        "path",
        type=str,
        help="Path to JSONL file to validate"
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=3,
        help="Minimum number of questions in followup list (default: 3)"
    )
    parser.add_argument(
        "--bullet-style",
        type=str,
        choices=["dash", "asterisk", "either"],
        default="either",
        help="Bullet style for followup list (default: either)"
    )
    parser.add_argument(
        "--no-require-question-mark",
        action="store_true",
        help="Allow followup items without question marks"
    )
    parser.add_argument(
        "--allow-extra-text",
        action="store_true",
        help="Allow extra text after followup list"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Stop after N total errors (default: 50)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, no per-row errors"
    )
    
    return parser


def validate_row(row: dict, *, min_questions: int, bullet_style: str, require_question_mark: bool, forbid_extra_text: bool) -> list[str]:
    """
    Validate a single JSONL row.
    
    Returns a list of error strings. Empty list means row is valid.
    """
    errors = []
    
    # Check if row is a dict
    if not isinstance(row, dict):
        return ["Row is not a JSON object"]
    
    # Check for messages key
    if "messages" not in row:
        return ["Missing key: messages"]
    
    # Check messages is a list and not empty
    if not isinstance(row["messages"], list):
        return ["messages is not a list"]
    
    if len(row["messages"]) == 0:
        return ["messages list is empty"]
    
    # Schema validation via Pydantic
    try:
        parsed = ChatRequest(**row)
    except ValidationError as e:
        for error in e.errors():
            errors.append(f"Schema error: {error['msg']} at {error['loc']}")
        return errors
    
    # Find last assistant message
    assistant_text = None
    for m in reversed(row["messages"]):
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip() != "":
                assistant_text = content.strip()
                break
    
    if assistant_text is None:
        return ["No valid assistant message found"]
    
    # Check for numbered lists
    pattern = re.compile(r"^\s*\d+[\.\)]\s+")
    for line in assistant_text.splitlines():
        if pattern.match(line):
            errors.append(f"Numbered list format not allowed: '{line}'")
    
    # Validate assistant content as followup bullets
    result = validate_followup_list(
        text=assistant_text,
        min_questions=min_questions,
        bullet_style=bullet_style,
        require_question_mark=require_question_mark,
        forbid_extra_text=forbid_extra_text,
    )
    
    if not result.ok:
        errors.append(f"Followup format invalid: {'; '.join(result.errors)}")
    
    return errors

    


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point for JSONL validator.
    
    Returns:
        0 if all rows valid
        1 if any validation errors
        2 if file/parsing problem
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Resolve path
    path = Path(args.path)
    if not path.exists() or not path.is_file():
        print(f"Error: File not found or is not a file: {args.path}", file=sys.stderr)
        return 2
    
    # Initialize counters
    row_idx = 0
    bad_rows = 0
    total_errors = 0
    
    # Invert the no-require-question-mark flag
    require_question_mark = not args.no_require_question_mark
    forbid_extra_text = not args.allow_extra_text
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row_idx += 1
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Try to parse JSON
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    bad_rows += 1
                    total_errors += 1
                    if not args.quiet:
                        print(f"Row {row_idx}: JSON parse error: {e}", file=sys.stderr)
                    if total_errors >= args.max_errors:
                        break
                    continue
                
                # Validate row
                errors = validate_row(
                    row,
                    min_questions=args.min_questions,
                    bullet_style=args.bullet_style,
                    require_question_mark=require_question_mark,
                    forbid_extra_text=forbid_extra_text,
                )
                
                if errors:
                    bad_rows += 1
                    total_errors += len(errors)
                    if not args.quiet:
                        for error in errors:
                            print(f"Row {row_idx}: {error}", file=sys.stderr)
                    if total_errors >= args.max_errors:
                        break
    
    except IOError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 2
    
    # Print summary
    print(f"Processed {row_idx} rows: {bad_rows} invalid, {total_errors} total errors")
    
    return 0 if bad_rows == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
