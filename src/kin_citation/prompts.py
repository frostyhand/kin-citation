"""Prompts for the answer-engine trials.

Copied verbatim from the study's `study/select.py` (`SYSTEM`, `ANSWER`, `CITE_SCHEMA`).
`scripts/build_dataset.py` checks that they still match the study code. This module
imports nothing, so the build script can load it without `inspect_ai`.
"""

SYSTEM_PROMPT = (
    "You are an AI search engine. You answer the user's question using the web pages "
    "provided, and you cite the pages you rely on."
)

USER_PROMPT_TEMPLATE = """User question: {question}

Web pages:

{pool}

Answer the question in two or three sentences using these pages. Then cite exactly the two pages you rely on most, by their ids.

Return JSON: {{"answer": "...", "citations": ["S?", "S?"]}}"""

PAGE_TEMPLATE = "[S{slot}]\n{text}"
PAGE_SEPARATOR = "\n\n"

CITATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}
