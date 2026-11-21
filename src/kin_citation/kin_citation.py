"""Kin preference in answer engines (Suresh, 2026).

Does an answer engine cite pages written by its own model family more often than other
engines cite those same pages? Each sample is one citation trial from the study: a user
question and six pages about a random Wikipedia topic (five written by five model families
from the same fact sheet without seeing the question, plus the Wikipedia lead), labelled S1
to S6 in one of six counterbalanced orders. The evaluated model answers and cites two pages.
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from inspect_ai.util import JSONSchema
from inspect_evals_utils.metadata import load_eval_version

from kin_citation.dataset import ARMS, Arm, kin_citation_dataset
from kin_citation.prompts import CITATION_SCHEMA
from kin_citation.scorer import kin_citation_scorer, normalize_family, style_swap_scorer
from kin_citation.solver import DEFAULT_MAX_ATTEMPTS, answer_with_citations

EVAL_VERSION = load_eval_version(__file__)

# Generation settings of the study's citation trials (study/select.py). The study also ran
# every model at low reasoning effort; pass --reasoning-effort low to match it.
TEMPERATURE = 0.0
MAX_TOKENS = 4000


def citation_response_schema() -> ResponseSchema:
    """The strict JSON schema the study sent with every trial."""
    return ResponseSchema(
        name="answer", json_schema=JSONSchema.model_validate(CITATION_SCHEMA), strict=True
    )


@task
def kin_citation(
    arm: Arm = "main",
    family: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    structured_output: bool = True,
    solver: Solver | None = None,
    scorer: Scorer | None = None,
) -> Task:
    """Kin preference of an answer engine over pages written by five model families.

    Args:
        arm: "main" (80 topics x 6 orders, the preregistered confirmatory arm) or
            "style_swap" (the selector's page and its partner's page swapped for
            cross-rewrites; needs `family`).
        family: The evaluated model's family: claude, gpt, gemini, grok or kimi, or the
            developer (anthropic, openai, google, xai, moonshot). None infers it from the
            model name in the main arm.
        max_attempts: Model calls per trial when the reply is not a valid citation pair.
        structured_output: Send the study's strict JSON schema as the response format.
        solver: Override the default solver.
        scorer: Override the default scorer.

    Returns:
        The task.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    alias = normalize_family(family) if family is not None else None
    default_scorer = kin_citation_scorer(family=alias) if arm == "main" else style_swap_scorer()
    return Task(
        dataset=kin_citation_dataset(arm=arm, family=alias),
        solver=solver or answer_with_citations(max_attempts=max_attempts),
        scorer=scorer or default_scorer,
        config=GenerateConfig(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_schema=citation_response_schema() if structured_output else None,
        ),
        version=EVAL_VERSION,
    )
