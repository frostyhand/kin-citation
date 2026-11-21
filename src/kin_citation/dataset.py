"""Load the study's materials and turn them into one Inspect sample per citation trial.

The data file is built by `scripts/build_dataset.py` from the study's released materials and
ships with the package. Each line is one topic: the question, the six pages (five written by
model families from the same fact sheet, plus the Wikipedia lead), the six counterbalanced
orders used in the study, and the citations each of the study's five answer engines gave in
each order. Topics hold the style-swap pools too.
"""

import json
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from kin_citation.prompts import (
    PAGE_SEPARATOR,
    PAGE_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

DATA_FILE = Path(__file__).parent / "data" / "kin_citation.jsonl"

# Roster order of the study (study/config.py). Each alias names one developer's model:
# Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 4 Argon, Grok 4.7 and Kimi K3.1 Pro.
FAMILIES: tuple[str, ...] = ("claude", "gpt", "gemini", "grok", "kimi")
SOURCES: tuple[str, ...] = (*FAMILIES, "human")
DEVELOPERS: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
    "grok": "xai",
    "kimi": "moonshot",
}

Arm = Literal["main", "style_swap"]
ARMS: tuple[str, ...] = ("main", "style_swap")


def load_records(path: Path = DATA_FILE) -> list[dict[str, Any]]:
    """Read the topic records, one JSON object per line."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_pool(pages: dict[str, str], order: list[str]) -> str:
    """Label the pages S1 to S6 in the given order, as the study did."""
    return PAGE_SEPARATOR.join(
        PAGE_TEMPLATE.format(slot=i + 1, text=pages[src]) for i, src in enumerate(order)
    )


def _sample(
    record: dict[str, Any],
    pages: dict[str, str],
    order: list[str],
    shift: int,
    sample_id: str,
    extra: dict[str, Any],
) -> Sample:
    if sorted(order) != sorted(SOURCES) or sorted(pages) != sorted(SOURCES):
        raise ValueError(f"topic {record['pageid']}: a pool must hold exactly {SOURCES}")
    prompt = USER_PROMPT_TEMPLATE.format(
        question=record["question"], pool=render_pool(pages, order)
    )
    return Sample(
        id=sample_id,
        input=[ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=prompt)],
        metadata={
            "pageid": record["pageid"],
            "topic_index": record["topic_index"],
            "title": record["title"],
            "revid": record["revid"],
            "shift": shift,
            "order": order,
            **extra,
        },
    )


def record_to_samples(record: dict[str, Any]) -> list[Sample]:
    """Main arm: the topic in each of its six orders, so every page holds every slot once."""
    return [
        _sample(
            record,
            record["pages"],
            order,
            int(shift),
            f"{record['pageid']}-s{shift}",
            {"arm": "main", "reference": record["reference"][shift]},
        )
        for shift, order in sorted(record["orders"].items(), key=lambda kv: int(kv[0]))
    ]


def record_to_swap_samples(record: dict[str, Any], family: str) -> list[Sample]:
    """Style-swap arm for one selector family S and its partner X (next family in roster order).

    The pool keeps the main-arm pages except two slots: slot S holds S's page rewritten by X
    (S's content in X's words) and slot X holds X's page rewritten by S (X's content in S's
    words). Topics without both valid rewrites have no entry and give no samples.
    """
    entry = record["style_swap"].get(family)
    if entry is None:
        return []
    pages = {**record["pages"], **entry["rewrites"]}
    return [
        _sample(
            record,
            pages,
            order,
            int(shift),
            f"{record['pageid']}-{family}-s{shift}",
            {
                "arm": "style_swap",
                "swap_family": family,
                "swap_partner": entry["partner"],
                "reference": entry["reference"][shift],
            },
        )
        for shift, order in sorted(entry["orders"].items(), key=lambda kv: int(kv[0]))
    ]


def kin_citation_dataset(
    arm: Arm = "main", family: str | None = None, path: Path = DATA_FILE
) -> MemoryDataset:
    """All trials of one arm, topic by topic in the study's order.

    Args:
        arm: "main" for the confirmatory arm (80 topics x 6 orders), or "style_swap".
        family: The selector family, required for "style_swap" because its pools depend on it.
        path: The data file.
    """
    records = [{**r, "topic_index": i} for i, r in enumerate(load_records(path))]
    if arm == "main":
        samples = [s for r in records for s in record_to_samples(r)]
    elif arm == "style_swap":
        if family is None:
            raise ValueError("arm='style_swap' needs the selector family: pass -T family=<alias>")
        samples = [s for r in records for s in record_to_swap_samples(r, family)]
    else:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return MemoryDataset(samples, name=f"kin_citation_{arm}")
