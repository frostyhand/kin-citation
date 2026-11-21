# Kin Citation: Kin Preference in Generative Answer Engines

Does an answer engine cite web pages written by its own model family more often than other engines cite those same pages? This is an [Inspect AI](https://inspect.aisi.org.uk/) implementation of the preregistered study [Do Answer Engines Cite Their Own Kind? A Controlled Test of Kin Preference in Generative Search](https://ssrn.com/abstract=7710958) (Suresh, 2026). It runs the study's exact materials and prompts against any model and reports the study's estimand, kin preference, together with the citation shares and the position effect.

<!-- Contributors: Automatically Generated -->
Contributed by [@frostyhand](https://github.com/frostyhand)
<!-- /Contributors: Automatically Generated -->

## Overview

Earlier work on whether language models favour text written by models like themselves disagrees, partly because authorship was confounded with how well a page answered the question. This eval removes that confound by construction:

- 80 random English Wikipedia topics. For each, a model outside the roster extracted a fixed sheet of atomic facts and one user question that needs at least two of them.
- Five frontier models from five developers (Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 4 Argon, Grok 4.7 and Kimi K3.1 Pro) each wrote one page per topic from the fact sheet alone, without seeing the question. A page was accepted only if it contained every key term of every fact and its length was 0.5 to 1.5 times the Wikipedia lead. The Wikipedia lead is the sixth, human page.
- The evaluated model acts as the answer engine. It sees the question and the six pages labelled S1 to S6, answers in two or three sentences and cites exactly the two pages it relies on most. Each topic is shown in six orders (a seeded base order and its cyclic shifts), so every page holds every slot once per topic.

Because every page carries the same facts and none was written for the question, a difference in how often the evaluated model cites its own family's page cannot come from query alignment. Because the study's five engines judged the same pages in the same orders, the eval can also remove page quality: kin preference compares the evaluated model with how the other engines cite its family's page.

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval kin_citation/kin_citation --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from kin_citation import kin_citation
eval(kin_citation)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval kin_citation --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval kin_citation/kin_citation --limit 10
uv run inspect eval kin_citation/kin_citation --max-connections 10
uv run inspect eval kin_citation/kin_citation --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `kin_citation`

- `arm` (Literal['main', 'style_swap']): "main" (80 topics x 6 orders, the preregistered confirmatory arm) or "style_swap" (the selector's page and its partner's page swapped for cross-rewrites; needs `family`). (default: `'main'`)
- `family` (str | None): The evaluated model's family: claude, gpt, gemini, grok or kimi, or the developer (anthropic, openai, google, xai, moonshot). None infers it from the model name in the main arm. (default: `None`)
- `max_attempts` (int): Model calls per trial when the reply is not a valid citation pair. (default: `2`)
- `structured_output` (bool): Send the study's strict JSON schema as the response format. (default: `True`)
- `solver` (Solver | None): Override the default solver. (default: `None`)
- `scorer` (Scorer | None): Override the default scorer. (default: `None`)
<!-- /Parameters: Automatically Generated -->

### Choosing the family

Kin is defined by developer. The scorer infers the evaluated model's family from its Inspect model name: the model id is checked first (`claude`, `gemini` or `gemma`, `grok`, `kimi`, `gpt` or `o<digit>`), then the provider and organisation segments (`anthropic`, `openai`, `google`, `x-ai`, `moonshotai`), skipping routers such as `openrouter` and `openai-api`. So `openrouter/openai/gpt-5-nano` is scored against the GPT-6.1 Sol page and `anthropic/claude-haiku-5-5` against the Claude Sonnet 5.5 page. Pass `-T family=<alias>` to override the guess. A model from any other developer still gets every citation share and the position metric, but its kin metrics are `nan` because no page in the pool was written by its family.

## Dataset

The data file `src/kin_citation/data/kin_citation.jsonl` ships with the package and is built deterministically by `scripts/build_dataset.py` from the study's materials (the script checks that the prompts here match the study code). The study repository itself is not public yet, so the shipped data file is the reference copy that the eval uses. It holds one line per topic: the Wikipedia page id, revision id and title, the question, the six pages, the six orders, and the citations each of the study's five answer engines gave in each order.

| Arm | Samples | What one sample is |
| --- | --- | --- |
| `main` (default) | 480 | one topic in one of its six orders (80 topics x 6 orders) |
| `style_swap` | 111, 114, 117, 81, 90 | one pool in one of three orders, for `family` claude, gpt, gemini, grok and kimi respectively |

Samples are ordered topic by topic in the study's order, so `--limit 6` runs one complete, position-balanced topic. Sample ids are `<pageid>-s<shift>` (main) and `<pageid>-<family>-s<shift>` (style swap).

**Style-swap arm.** On the first 40 topics, for a selector family S and its partner X (the next family in roster order: claude, gpt, gemini, grok, kimi, back to claude), X rewrote S's page and S rewrote X's page, keeping every fact and key term. The pool replaces the two originals with the two rewrites. If kin preference follows surface style, S should cite the page it rewrote (X's content in S's words) more than the page X rewrote (S's content in X's words). Pools exist only where both rewrites passed the fact check, which is why the sample counts differ by family. This arm needs `-T family=...` because the pool depends on the selector.

## Scoring

The solver sends the study's system prompt, the question and the pool, with the study's strict JSON schema as the response format (`{"answer": ..., "citations": [...]}`), at temperature 0 and up to 4,000 output tokens. A reply is valid when it parses as a JSON object and its `citations` are exactly two distinct ids among S1 to S6 (brackets, case and surrounding whitespace are tolerated, as in the study). An invalid reply is discarded and the same prompt is sent once more (`max_attempts=2`); a trial still invalid scores `valid = 0` with reason `invalid_response_format` and is left out of every share, as the preregistration specifies.

Each sample's score is a dictionary. Per-key metrics are the mean and a topic-clustered standard error.

| Key | Meaning | Chance |
| --- | --- | --- |
| `valid` | share of trials with a valid citation pair | |
| `cited_<source>` | share of valid trials citing that page (`claude`, `gpt`, `gemini`, `grok`, `kimi`, `human`) | 1/3 |
| `kin_cited` | share of valid trials citing the evaluated model's own family's page | 1/3 |
| `kin_minus_other_llm` | own family's page share minus the mean share of the other four model pages (raw, within this model) | 0 |
| `human_minus_llm` | Wikipedia page share minus the mean share of the five model pages | 0 |
| `first_slot_share` | share of all citations that go to slot S1 | 1/6 |
| `own_words_cited`, `own_content_cited`, `own_words_minus_own_content` | style-swap arm only: the page the selector rewrote, the page its partner rewrote, and their difference | |

**Kin preference (headline metric).** `kin_minus_other_llm` is easy to read but confounded: if one family's pages are simply better, every engine cites them more. The study's estimand removes this. Let C be the 5 x 5 matrix of citation shares with one row per engine and one column per model-written page. The evaluated model fills its own family's row; the study's other four engines fill the remaining rows, computed on exactly the samples that were run. Double centring gives R = C minus row means minus column means plus the grand mean, and kin preference is K = R[f, f] for the evaluated family f. The column effect absorbs everything about a page that makes all engines cite it more or less; the row effect absorbs the evaluated model's general habits. K is positive only if the model cites its own family's page more than the other engines cite that same page, beyond what its own tendencies predict. `kin_preference_ci_lower` and `kin_preference_ci_upper` are the 95% percentile interval from 10,000 bootstrap resamples of topics (seed 20261115), with all trials of a topic resampled together.

For one of the study's own engines, the eval reproduces the published estimate exactly: replaying each engine's recorded replies through this scorer gives the study's K and its interval to twelve decimal places (`tests/test_kin_citation.py::test_kin_preference_reproduces_the_study`).

## Reference results

From the study (Suresh, 2026): each engine run on the full main arm (480 trials, low reasoning effort, through OpenRouter, November 2026). Values are percentage points or percentages.

| Engine | Kin preference K (95% CI) | Own family's page cited | First-slot share | Valid |
| --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`) | -4.6 (-8.6, -0.5) | 28.4% | 24.5% | 99.8% |
| GPT-6.1 Sol (`openai/gpt-6.1-sol`) | 8.8 (5.4, 12.3) | 28.5% | 31.6% | 100% |
| Gemini 4 Argon (`google/gemini-4-argon`) | 4.6 (2.0, 7.3) | 23.1% | 40.5% | 100% |
| Grok 4.7 (`x-ai/grok-4.7`) | 1.8 (-0.1, 3.8) | 14.9% | 44.7% | 99.4% |
| Kimi K3.1 Pro (`moonshotai/kimi-k3.1-pro`) | -2.3 (-4.5, -0.1) | 17.7% | 36.6% | 100% |

Pooled over the five engines, K is 1.7 points (95% CI 0.3 to 3.1). Every engine cites the Wikipedia lead in 93% to 98% of trials, so the model pages compete for the remaining citation, and 35.6% of all citations go to slot S1. In the style-swap arm only Kimi K3.1 Pro cites its own words more than its own content (17.8 points, 95% CI 5.1 to 31.2).

To reproduce a row, run the same model through OpenRouter with the study's reasoning setting:

```bash
uv run inspect eval kin_citation/kin_citation --model openrouter/openai/gpt-6.1-sol --reasoning-effort low
```

A rerun will not match the published replies token for token, because providers do not guarantee determinism at temperature 0 and OpenRouter may route a model to different providers.

## Evaluation report

No full run has been made with this implementation yet; the table above comes from the study's own harness, whose recorded replies this eval reproduces exactly. A smoke run on five samples (one topic, five of its six orders) confirmed the pipeline end to end:

| Model | Command | Samples | Valid | Notes |
| --- | --- | --- | --- | --- |
| `openrouter/openai/gpt-5-nano` | `uv run inspect eval kin_citation/kin_citation --model openrouter/openai/gpt-5-nano --limit 5` | 5 | 100% | every reply parsed on the first attempt; 4 of 5 cited S1 and S2 |

Five samples from one topic are far too few for any estimate, so no kin preference is reported for this run. Version 1-A, November 2026.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| The evaluated model's own row is combined with the study's recorded replies for the other four engines, instead of five engines run together. | Inspect evaluates one model per run. | None for the study's own engines (reproduced exactly). For a new model, K is relative to the study's four other engines. |
| A reply whose JSON is not an object (for example a bare list) is invalid. | The study code assumed an object; such a reply never occurred there. | None on the study's data. |
| Reasoning effort is not fixed by the task; the study used low effort for every model. | Not every model supports it; pass `--reasoning-effort low` to match. | Can change citations for reasoning models. |
| The retry resends the same prompt inside one sample, and the log keeps both calls. | Inspect keeps the transcript per sample. | None. |

The self-recognition and sibling arms of the study are not included: self-recognition asks a different question (which page the model wrote) and the sibling arm only applies to two families.

## Limitations

These follow the paper. Pages were written without the question, which removes query alignment but makes them less like real web pages. Each trial shows six pages and asks for two citations in one prompt format, while production engines retrieve more pages and use their own system prompts. The Wikipedia lead carries more content than the model pages, so `human_minus_llm` does not isolate authorship. The fact check is lexical. With 80 topics, per-model intervals are a few points wide, and the style-swap pools are smaller still. The pages were written by specific model versions in November 2026; for a later model of the same family, kin preference measures preference for that earlier sibling's pages.

## Architecture

```text
data/kin_citation.jsonl  ->  dataset.py: one Sample per (topic, order)
                               system prompt + question + six pages S1..S6
                                         |
                             solver.py: generate with JSON schema,
                               resend once if the reply is not two distinct ids
                                         |
                             scorer.py: map ids to pages, per-sample shares;
                               kin_preference metric adds the study's other
                               four engines on the same samples, double centres,
                               topic bootstrap
```

No sandbox, tools or grader model are used.

## Getting started

```bash
uv sync
uv run pre-commit install
uv run inspect eval kin_citation/kin_citation --model openrouter/openai/gpt-5-nano --limit 6
uv run inspect view
```

Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time. A full main-arm run is 480 calls of about 1,400 input tokens each.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest; no test calls a model API
```

The tests cover the dataset (counts, counterbalancing, prompt rendering), citation parsing, family inference, per-sample values, the metrics through Inspect's epoch reducer, end-to-end runs on `mockllm/model`, and the exact reproduction of the study's estimates from its recorded replies.

## Layout

```text
src/kin_citation/kin_citation.py   the task
src/kin_citation/dataset.py        samples from the data file
src/kin_citation/solver.py         generate with one retry
src/kin_citation/scorer.py         parsing, scorers, kin_preference metric
src/kin_citation/prompts.py        the study's prompts and schema, verbatim
src/kin_citation/data/             the data file
scripts/build_dataset.py           rebuilds the data file from the study materials
tests/test_kin_citation.py         the tests
```

## Data and licences

Code: MIT. Pages written for the study, the questions and the recorded replies: CC BY 4.0. The Wikipedia leads (the `human` page of each topic) are from English Wikipedia under CC BY-SA 4.0; each topic records the page id and revision id, so the lead of topic `revid` is at `https://en.wikipedia.org/w/index.php?oldid=<revid>`. The data file as a whole is therefore distributed under CC BY-SA 4.0.

## Citation

```bibtex
@misc{suresh2026kin,
  title  = {Do Answer Engines Cite Their Own Kind? A Controlled Test of Kin Preference in Generative Search},
  author = {Suresh, Tejaswi},
  year   = {2026},
  note   = {SSRN preprint 7710958},
  url    = {https://ssrn.com/abstract=7710958}
}
```

## Changelog

### [1-A] 2026-11

- Initial implementation of the main and style-swap arms.
