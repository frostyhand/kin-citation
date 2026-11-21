"""Generate a cited answer, retrying once when the reply is not a valid citation pair."""

from inspect_ai.solver import Generate, Solver, TaskState, solver

from kin_citation.scorer import parse_citations

DEFAULT_MAX_ATTEMPTS = 2  # the study retried an invalid reply once


@solver
def answer_with_citations(max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Solver:
    """Ask the answer engine for a cited answer, as study/select.py `run_trial` did.

    A reply that does not cite exactly two distinct ids among S1 to S6 is discarded and the
    same prompt is sent again, up to `max_attempts` calls in all. The last reply is scored.

    Args:
        max_attempts: Model calls per trial, at least 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = list(state.messages)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            state.messages = list(prompt)
            state = await generate(state)
            if parse_citations(state.output.completion, state.metadata["order"]) is not None:
                break
        state.metadata["attempts"] = attempts
        return state

    return solve
