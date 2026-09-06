from __future__ import annotations

import re


def normalise(text: str) -> str:
    return " ".join(
        text.casefold().split()
    )


def contains_keyword(
    text: str,
    keyword: str,
) -> bool:
    normalised_text = normalise(
        text
    )

    normalised_keyword = normalise(
        keyword
    )

    if not normalised_keyword:
        return False

    if (
        " " in normalised_keyword
        or "-" in normalised_keyword
    ):
        return (
            normalised_keyword
            in normalised_text
        )

    return bool(
        re.search(
            rf"(?<!\w)"
            rf"{re.escape(normalised_keyword)}"
            rf"(?!\w)",
            normalised_text,
        )
    )
