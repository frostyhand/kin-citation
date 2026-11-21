import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai import eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Score
from pytest_inspect_evals.assertions import assert_eval_success
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

from kin_citation import kin_citation
from kin_citation.dataset import (
    FAMILIES,
    SOURCES,
    kin_citation_dataset,
    load_records,
    record_to_samples,
    render_pool,
)
from kin_citation.kin_citation import EVAL_VERSION
from kin_citation.prompts import SYSTEM_PROMPT
from kin_citation.scorer import (
    MAIN_KEYS,
    SWAP_KEYS,
    infer_family,
    kin_preference,
    main_values,
    normalize_family,
    parse_citations,
    style_swap_scorer,
    swap_values,
)
from kin_citation.solver import answer_with_citations

ORDER = ["gemini", "human", "claude", "grok", "kimi", "gpt"]

# From the study's results/summary.json (`kin`): K_s and its 95% topic-bootstrap interval
# for each of the study's answer engines, recomputed here from the shipped trial records.
STUDY_KIN = {
    "claude": (-0.046172750270260854, -0.08599884091486015, -0.00474922386702686),
    "gpt": (0.08804588415768347, 0.05420071739472962, 0.12334930331197147),
    "gemini": (0.04592678776976841, 0.020250000000000018, 0.07333337819443903),
    "grok": (0.018133308605016524, -0.0012505781890531332, 0.037745267495893936),
    "kimi": (-0.023220708980536853, -0.04515620344972078, -0.0009595507347743448),
}
# results/summary.json `swap_ci`: per selector, share citing its own words minus its own content.
STUDY_SWAP = {
    "claude": (0.02702702702702703, 111),
    "gpt": (-0.02631578947368421, 114),
    "gemini": (-0.02564102564102564, 117),
    "grok": (0.025, 80),
    "kimi": (0.17777777777777778, 90),
}


def kin(scores: list[SampleScore]) -> dict[str, float]:
    result = cast(Any, kin_preference())(scores)
    assert isinstance(result, dict)
    return result


def user_text(sample: Sample) -> str:
    assert isinstance(sample.input, list)
    return sample.input[1].text


def meta(sample: Sample) -> dict[str, Any]:
    assert sample.metadata is not None
    return sample.metadata


def reply(*ids: str) -> str:
    return json.dumps({"answer": "An answer.", "citations": list(ids)})


def replay_scores(family: str) -> list[SampleScore]:
    """The study engine's own citations on every main-arm trial, scored as this eval would."""
    return [
        SampleScore(
            score=Score(
                value=main_values(meta(s)["reference"][family], meta(s)["order"], family),
                metadata={"family": family},
            ),
            sample_id=s.id,
            sample_metadata=s.metadata,
        )
        for s in kin_citation_dataset()
    ]


# Dataset


def test_record_to_samples_with_a_real_record() -> None:
    record = {**load_records()[0], "topic_index": 0}
    samples = record_to_samples(record)
    assert [s.id for s in samples] == [f"12776606-s{k}" for k in range(6)]
    first = samples[0]
    assert first.metadata is not None
    assert first.metadata["order"] == ORDER
    assert first.metadata["reference"]["gpt"] == ["gemini", "human"]
    assert isinstance(first.input, list)
    assert first.input[0].text == SYSTEM_PROMPT
    text = user_text(first)
    assert text.startswith("User question: What was Polirom's first published book")
    assert text.index("[S1]\n" + record["pages"]["gemini"][:40]) > 0
    assert text.endswith('Return JSON: {"answer": "...", "citations": ["S?", "S?"]}')


def test_main_arm_is_counterbalanced() -> None:
    ds = kin_citation_dataset()
    assert len(ds) == 480
    assert len({s.id for s in ds}) == 480
    by_topic: dict[int, list[list[str]]] = {}
    for s in ds:
        assert s.metadata is not None
        by_topic.setdefault(s.metadata["pageid"], []).append(s.metadata["order"])
    assert len(by_topic) == 80
    for orders in by_topic.values():
        for slot in range(6):
            assert sorted(o[slot] for o in orders) == sorted(SOURCES)


def test_style_swap_arm_pools() -> None:
    counts = {f: len(kin_citation_dataset(arm="style_swap", family=f)) for f in FAMILIES}
    # Pools with both rewrites valid, x 3 orders; one Grok trial was invalid in the study (80 valid).
    assert counts == {"claude": 111, "gpt": 114, "gemini": 117, "grok": 81, "kimi": 90}
    main = {s.id: s for s in kin_citation_dataset()}
    s = kin_citation_dataset(arm="style_swap", family="gpt")[0]
    assert s.metadata is not None
    assert (s.metadata["swap_family"], s.metadata["swap_partner"]) == ("gpt", "gemini")
    # The two swapped slots differ from the main pool; the other four pages are unchanged.
    main_text = user_text(main[f"{s.metadata['pageid']}-s0"])
    swap_text = user_text(s)
    assert main_text != swap_text
    record = load_records()[0]
    for src in ("claude", "grok", "kimi", "human"):
        assert record["pages"][src] in swap_text


def test_render_pool_labels_in_order() -> None:
    pages = {src: src.upper() for src in SOURCES}
    assert render_pool(pages, ORDER).split("\n\n")[:2] == ["[S1]\nGEMINI", "[S2]\nHUMAN"]


# Parsing and family


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (reply("S1", "S6"), ["gemini", "gpt"]),
        (reply("[S2]", " s3 "), ["human", "claude"]),
        ("```json\n" + reply("S4", "S5") + "\n```", ["grok", "kimi"]),
        ("Here it is: " + reply("S1", "S2") + " done", ["gemini", "human"]),
        (reply("S1", "S1"), None),
        (reply("S1"), None),
        (reply("S1", "S2", "S3"), None),
        (reply("S1", "S7"), None),
        (reply("S1", "page 2"), None),
        ('["S1", "S2"]', None),
        ('{"answer": "x", "citations": "S1, S2"}', None),
        ('{"answer": "x"}', None),
        ("no json at all", None),
        ("", None),
    ],
)
def test_parse_citations(completion: str, expected: list[str] | None) -> None:
    assert parse_citations(completion, ORDER) == expected


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("anthropic/claude-sonnet-5-5", "claude"),
        ("bedrock/anthropic.claude-haiku-5-5", "claude"),
        ("openai/gpt-5-nano", "gpt"),
        ("openai/o4-mini", "gpt"),
        ("openrouter/openai/gpt-6.1-sol", "gpt"),
        ("together/openai/gpt-oss-120b", "gpt"),
        ("google/gemini-4-argon", "gemini"),
        ("hf/google/gemma-3-4b-it", "gemini"),
        ("grok/grok-4.7", "grok"),
        ("openrouter/x-ai/grok-4.7", "grok"),
        ("openrouter/moonshotai/kimi-k3.1-pro", "kimi"),
        ("openai-api/deepseek/deepseek-chat", None),
        ("openrouter/meta-llama/llama-4-maverick", None),
        ("mockllm/model", None),
    ],
)
def test_infer_family(model: str, family: str | None) -> None:
    assert infer_family(model) == family


def test_normalize_family() -> None:
    assert normalize_family("OpenAI") == "gpt"
    assert normalize_family("x-ai") == "grok"
    assert normalize_family("kimi") == "kimi"
    with pytest.raises(ValueError, match="unknown family"):
        normalize_family("meta")


# Per-trial values


def test_main_values_valid_trial() -> None:
    v = main_values(["gpt", "human"], ORDER, "gpt")
    assert set(v) == set(MAIN_KEYS)
    assert v["valid"] == 1.0
    assert v["cited_gpt"] == v["cited_human"] == 1.0
    assert v["cited_claude"] == 0.0
    assert v["kin_cited"] == 1.0
    assert v["kin_minus_other_llm"] == 1.0
    assert v["human_minus_llm"] == pytest.approx(0.8)
    assert v["first_slot_share"] == 0.0
    v = main_values(["gemini", "claude"], ORDER, "gpt")
    assert v["kin_cited"] == 0.0
    assert v["kin_minus_other_llm"] == -0.5
    assert v["first_slot_share"] == 0.5


def test_main_values_unknown_family_and_invalid() -> None:
    v = main_values(["gpt", "human"], ORDER, None)
    assert math.isnan(v["kin_cited"])
    assert math.isnan(v["kin_minus_other_llm"])
    assert v["cited_gpt"] == 1.0
    invalid = main_values(None, ORDER, "gpt")
    assert set(invalid) == set(MAIN_KEYS)
    assert invalid["valid"] == 0.0
    assert all(math.isnan(x) for k, x in invalid.items() if k != "valid")


def test_swap_values() -> None:
    v = swap_values(["gemini", "human"], ORDER, "gpt", "gemini")
    assert set(v) == set(SWAP_KEYS)
    assert (v["own_words_cited"], v["own_content_cited"], v["own_words_minus_own_content"]) == (
        1,
        0,
        1,
    )
    invalid = swap_values(None, ORDER, "gpt", "gemini")
    assert invalid["valid"] == 0.0
    assert math.isnan(invalid["own_words_minus_own_content"])


# Metrics: the eval reproduces the study's estimates from the study's own trials


@pytest.mark.parametrize("family", FAMILIES)
def test_kin_preference_reproduces_the_study(family: str) -> None:
    result = kin(replay_scores(family))
    k, lo, hi = STUDY_KIN[family]
    assert isinstance(result, dict)
    assert result["kin_preference"] == pytest.approx(k, abs=1e-12)
    assert result["kin_preference_ci_lower"] == pytest.approx(lo, abs=1e-12)
    assert result["kin_preference_ci_upper"] == pytest.approx(hi, abs=1e-12)


def test_style_swap_replay_reproduces_the_study() -> None:
    for family, (diff, n) in STUDY_SWAP.items():
        values = []
        for s in kin_citation_dataset(arm="style_swap", family=family):
            assert s.metadata is not None
            cited = s.metadata["reference"]
            v = swap_values(cited, s.metadata["order"], family, s.metadata["swap_partner"])
            if v["valid"]:
                values.append(v["own_words_minus_own_content"])
        assert len(values) == n
        assert sum(values) / n == pytest.approx(diff)


def test_first_slot_share_matches_the_study() -> None:
    shares = []
    for family in FAMILIES:
        for s in replay_scores(family):
            assert isinstance(s.score.value, dict)
            x = float(cast(float, s.score.value["first_slot_share"]))
            if not math.isnan(x):
                shares.append(x)
    # results/summary.json position.share[0]: 35.6% of all citations go to slot S1.
    assert sum(shares) / len(shares) == pytest.approx(0.3555926544240401)


def test_kin_preference_is_nan_without_a_family_or_valid_trials() -> None:
    scores = replay_scores("gpt")[:6]
    no_family = [
        SampleScore(
            score=Score(value=s.score.value, metadata={"family": None}),
            sample_metadata=s.sample_metadata,
        )
        for s in scores
    ]
    assert all(math.isnan(x) for x in kin(no_family).values())
    invalid = [
        SampleScore(
            score=Score(value=main_values(None, ORDER, "gpt"), metadata={"family": "gpt"}),
            sample_metadata=s.sample_metadata,
        )
        for s in scores
    ]
    assert all(math.isnan(x) for x in kin(invalid).values())


def test_kin_preference_rejects_scalar_scores() -> None:
    scores = replay_scores("gpt")[:2]
    bad = [
        SampleScore(
            score=Score(value=1.0, metadata={"family": "gpt"}), sample_metadata=s.sample_metadata
        )
        for s in scores
    ]
    with pytest.raises(ValueError, match="dict score"):
        kin(bad)


def test_kin_preference_through_the_epoch_reducer() -> None:
    scores = replay_scores("gpt")[:24]  # four topics
    metadata = [s.sample_metadata or {} for s in scores]
    raw = [s.score for s in scores]
    assert_agreeing_epochs_change_nothing([kin_preference()], raw, metadata=metadata)
    # Two epochs that disagree average under `mean`: the evaluated row moves halfway.
    flipped = [
        Score(value=main_values(["human", "gpt"], m["order"], "gpt"), metadata={"family": "gpt"})
        for m in metadata
    ]
    both = run_metrics(
        [kin_preference()], [[a, b] for a, b in zip(raw, flipped, strict=True)], metadata=metadata
    )
    only_a = run_metrics([kin_preference()], [[a] for a in raw], metadata=metadata)
    only_b = run_metrics([kin_preference()], [[b] for b in flipped], metadata=metadata)
    # K is linear in the evaluated row, so the reduced K is the mean of the two.
    halfway = (
        float(cast(float, only_a["kin_preference"])) + float(cast(float, only_b["kin_preference"]))
    ) / 2
    assert both["kin_preference"] == pytest.approx(halfway)


# Task, solver and end to end


def test_version_comes_from_eval_yaml() -> None:
    assert EVAL_VERSION == "1-A"


def test_task_parameter_validation() -> None:
    with pytest.raises(ValueError, match="arm must be one of"):
        kin_citation(arm="sibling")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs the selector family"):
        kin_citation(arm="style_swap")
    with pytest.raises(ValueError, match="unknown family"):
        kin_citation(family="meta")
    with pytest.raises(ValueError, match="max_attempts"):
        kin_citation(max_attempts=0)


def test_end_to_end_with_mock_model(tmp_path: Path) -> None:
    # mockllm/model's default reply is not JSON, so every trial is invalid after one retry.
    [log] = eval(tasks=kin_citation(), model="mockllm/model", limit=3, log_dir=str(tmp_path))
    assert_eval_success(log, expected_sample_count=3)
    assert log.samples is not None
    for sample in log.samples:
        assert sample.scores is not None
        score = sample.scores["kin_citation_scorer"]
        assert score.reason == "invalid_response_format"
        assert sample.metadata["attempts"] == 2


def test_end_to_end_retry_then_valid_reply(tmp_path: Path) -> None:
    outputs = [
        ModelOutput.from_content(model="mockllm/model", content="I think S1 and S2."),
        ModelOutput.from_content(model="mockllm/model", content=reply("S6", "S2")),
    ]
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = eval(tasks=kin_citation(family="gpt"), model=model, limit=1, log_dir=str(tmp_path))
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.metadata["attempts"] == 2
    assert sample.scores is not None
    value = sample.scores["kin_citation_scorer"].value
    assert isinstance(value, dict)
    assert value["valid"] == 1.0
    assert value["kin_cited"] == 1.0
    assert value["cited_human"] == 1.0


def test_end_to_end_metrics_on_one_topic(tmp_path: Path) -> None:
    # The study's GPT engine replies, replayed for one topic: the eval's metrics match the replay.
    ds = kin_citation_dataset()
    outputs = []
    for s in list(ds)[:6]:
        assert s.metadata is not None
        cited = s.metadata["reference"]["gpt"]
        outputs.append(
            ModelOutput.from_content(
                model="mockllm/model",
                content=reply(*(f"S{s.metadata['order'].index(c) + 1}" for c in cited)),
            )
        )
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = eval(
        tasks=kin_citation(family="openai"),
        model=model,
        limit=6,
        max_connections=1,
        log_dir=str(tmp_path),
    )
    assert_eval_success(log, expected_sample_count=6)
    assert log.results is not None
    names = Counter(m for s in log.results.scores for m in s.metrics)
    assert names["kin_preference"] == 1
    expected = kin(replay_scores("gpt")[:6])
    got = {m.name: m.value for s in log.results.scores for m in s.metrics.values()}
    assert isinstance(expected, dict)
    assert got["kin_preference"] == pytest.approx(expected["kin_preference"])


def test_end_to_end_style_swap(tmp_path: Path) -> None:
    [log] = eval(
        tasks=kin_citation(arm="style_swap", family="kimi"),
        model="mockllm/model",
        limit=2,
        log_dir=str(tmp_path),
    )
    assert_eval_success(log, expected_sample_count=2)


def test_answer_with_citations_single_attempt(tmp_path: Path) -> None:
    task = kin_citation(solver=answer_with_citations(max_attempts=1))
    [log] = eval(tasks=task, model="mockllm/model", limit=1, log_dir=str(tmp_path))
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    assert log.samples[0].metadata["attempts"] == 1
    with pytest.raises(ValueError, match="max_attempts"):
        answer_with_citations(max_attempts=0)


def test_style_swap_scorer_reads_the_swapped_slots(tmp_path: Path) -> None:
    sample = kin_citation_dataset(arm="style_swap", family="gpt")[0]
    order = meta(sample)["order"]
    # Cite the partner's slot (gemini: Gemini's content in GPT's words) and the human page.
    content = reply(f"S{order.index('gemini') + 1}", f"S{order.index('human') + 1}")
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=content)],
    )
    task = kin_citation(arm="style_swap", family="gpt", scorer=style_swap_scorer())
    [log] = eval(tasks=task, model=model, limit=1, log_dir=str(tmp_path))
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    assert log.samples[0].scores is not None
    value = log.samples[0].scores["style_swap_scorer"].value
    assert isinstance(value, dict)
    assert value["own_words_cited"] == 1.0
    assert value["own_content_cited"] == 0.0
    assert value["own_words_minus_own_content"] == 1.0
