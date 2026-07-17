from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from poetry_reasoning.evaluation import judge  # noqa: E402


GATE_SYSTEMS = [
    "deepseek_v4_flash_thinking_max",
    "qwen36_plus_thinking_b2048",
    "qwen36_plus_thinking_b4096",
]


class FakeResponse:
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "{}", "reasoning_content": "audit reasoning"}}],
            "usage": {"completion_tokens": 7},
            "model": "gpt-5.5-actual",
            "system_fingerprint": "fp-test",
        }


def test_strict_json_accepts_one_complete_json_code_fence() -> None:
    assert judge.parse_strict_json('```json\n{"candidate_scores": [], "ranking": []}\n```') == {
        "candidate_scores": [],
        "ranking": [],
    }


@pytest.mark.parametrize(
    "text",
    [
        'preamble\n```json\n{"candidate_scores": [], "ranking": []}\n```',
        '```json\n{"candidate_scores": [], "ranking": []}\n```\ntrailing text',
        '```python\n{"candidate_scores": [], "ranking": []}\n```',
    ],
)
def test_strict_json_rejects_non_json_fence_or_surrounding_prose(text: str) -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        judge.parse_strict_json(text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def dataset_rows(count: int) -> list[dict]:
    return [
        {
            "record_id": f"r{i}",
            "source_id": f"source-{i % 2}",
            "original_zh": f"诗{i}",
            "translation_en": f"reference {i}",
            "language_pair": "zh-en",
        }
        for i in range(count)
    ]


def translation_rows(system_id: str, count: int) -> list[dict]:
    return [
        {
            "record_id": f"r{i}",
            "source_id": f"source-{i % 2}",
            "system_id": system_id,
            "status": "ok",
            "output_text": f"{system_id} translation {i}",
            "reasoning_content": "PRIVATE_REASONING_SENTINEL",
            "prompt_version": "understand_translate_v1",
        }
        for i in range(count)
    ]


def score_rows(count: int) -> list[dict]:
    required = {
        "comet": 0.5,
        "bertscore_f1": 0.6,
        "sacrebleu_sentence": 10.0,
        "chrfpp_sentence": 20.0,
        "ter_sentence": 30.0,
        "line_count_diff": 1,
        "length_ratio": 1.05,
    }
    return [
        {"record_id": f"r{i}", "system_id": system_id, "status": "ok", **required}
        for system_id in GATE_SYSTEMS
        for i in range(count)
    ]


def gate_artifacts(tmp_path: Path, count: int = 2) -> tuple[Path, list[tuple[str, Path]], Path]:
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    specs: list[tuple[str, Path]] = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        write_jsonl(path, translation_rows(system_id, count))
        specs.append((system_id, path))
    write_jsonl(scores_path, score_rows(count))
    return dataset_path, specs, scores_path


def valid_judge_payload(candidate_ids: list[str], ranking: list[str] | None = None) -> dict:
    return {
        "candidate_scores": [
            {
                "candidate_id": candidate_id,
                "scores": {dimension: 5.0 for dimension in judge.JUDGE_DIMENSIONS},
                "reasons": {dimension: "reason" for dimension in judge.JUDGE_DIMENSIONS},
                "average_score": 5.0,
                "major_errors": [],
            }
            for candidate_id in candidate_ids
        ],
        "ranking": ranking or candidate_ids,
        "ranking_reason": "reason",
    }


def private_mapping_row(record_id: str = "r0") -> dict:
    return {
        "record_id": record_id,
        "candidate_map": [
            {"candidate_id": "A", "system_id": "s1", "system_name": "s1"},
            {"candidate_id": "B", "system_id": "s2", "system_name": "s2"},
        ],
    }


def public_raw_row(record_id: str = "r0") -> dict:
    return {
        "record_id": record_id,
        "source_id": "source",
        "status": "ok",
        "judge_id": "gpt-test",
        "judge_model": "gpt-test-model",
        "judge_prompt_version": judge.JUDGE_PROMPT_VERSION,
        "parsed_response": valid_judge_payload(["A", "B"]),
    }


def test_gpt_client_uses_high_json_payload_without_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict = {}

    def fake_post(url, *, headers, json, timeout):
        sent.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setenv("JUDGE_GPT_API_KEY", "secret-key")
    monkeypatch.setenv("JUDGE_GPT_BASE_URL", "https://transit.example/v1")
    monkeypatch.setenv("JUDGE_GPT_MODEL", "gpt-5.5")
    monkeypatch.setenv("JUDGE_GPT_MAX_TOKENS", "1234")
    monkeypatch.setattr(judge.requests, "post", fake_post)

    result = judge.GPTJudgeClient().evaluate("judge this")

    assert sent["url"] == "https://transit.example/v1/chat/completions"
    assert sent["payload"] == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "judge this"}],
        "reasoning_effort": "high",
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1234,
    }
    assert "temperature" not in sent["payload"]
    assert "top_p" not in sent["payload"]
    assert result["content"] == "{}"
    assert result["reasoning_content"] == "audit reasoning"
    assert result["model"] == "gpt-5.5-actual"
    assert result["system_fingerprint"] == "fp-test"
    assert "messages" not in result["request_parameters"]
    assert "secret-key" not in json.dumps(result["request_parameters"])


def test_deepseek_client_payload_remains_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict = {}

    def fake_post(url, *, headers, json, timeout):
        sent["payload"] = json
        return FakeResponse()

    monkeypatch.setenv("JUDGE_DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv("JUDGE_DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("JUDGE_TEMPERATURE", "0")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "8192")
    monkeypatch.setenv("JUDGE_DISABLE_THINKING_PARAM", "1")
    monkeypatch.delenv("JUDGE_RESPONSE_FORMAT_JSON", raising=False)
    monkeypatch.setattr(judge.requests, "post", fake_post)

    judge.DeepSeekJudgeClient().evaluate("judge this")

    assert sent["payload"] == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "judge this"}],
        "temperature": 0.0,
        "max_tokens": 8192,
        "stream": False,
        "thinking": {"type": "disabled"},
    }


def test_ranking_rejects_duplicate_ids_even_when_set_matches() -> None:
    payload = valid_judge_payload(["A", "B"], ranking=["A", "B", "A"])

    with pytest.raises(ValueError, match="exactly once"):
        judge.validate_judge_json(payload, ["A", "B"])


def test_multiple_output_files_merge_latest_rows_in_path_order(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"system_id": "s", "record_id": "r", "output_text": "old"}])
    write_jsonl(second, [{"system_id": "s", "record_id": "r", "output_text": "new"}])

    latest = judge.latest_outputs_by_key([first, second])

    assert latest[("s", "r")]["output_text"] == "new"


def test_reused_mapping_preserves_all_20_by_7_candidate_ids_order_and_uses_only_output_text(tmp_path: Path) -> None:
    systems = [f"system-{i}" for i in range(7)]
    manifest = [{"record_id": f"r{i}", "source_id": "source", "source_zh": "诗", "reference_en": "ref"} for i in range(20)]
    mapping_rows = [
        {
            "record_id": record["record_id"],
            "candidate_map": [
                {"candidate_id": chr(65 + i), "system_id": system_id, "system_name": system_id}
                for i, system_id in enumerate(reversed(systems))
            ],
        }
        for record in manifest
    ]
    mapping_path = tmp_path / "private_mapping.jsonl"
    verification_path = tmp_path / "verification.json"
    write_jsonl(mapping_path, mapping_rows)
    outputs = {
        (system_id, record["record_id"]): {
            "system_id": system_id,
            "record_id": record["record_id"],
            "status": "ok",
            "output_text": f"VISIBLE {system_id} {record['record_id']}",
            "reasoning_content": "PRIVATE_REASONING_SENTINEL",
        }
        for record in manifest
        for system_id in systems
    }

    reused = judge.prepare_reused_mapping(
        manifest,
        outputs,
        systems,
        mapping_path,
        verification_path,
    )

    assert len(reused) == 20
    assert sum(len(candidates) for candidates in reused.values()) == 140
    assert [candidate["candidate_id"] for candidate in reused["r0"]] == list("ABCDEFG")
    assert [candidate["system_id"] for candidate in reused["r0"]] == list(reversed(systems))
    prompt = judge.build_judge_prompt(manifest[0], reused["r0"])
    assert "VISIBLE" in prompt
    assert "PRIVATE_REASONING_SENTINEL" not in prompt
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification == {
        "input_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "record_count": 20,
        "candidate_count": 140,
        "passed": True,
    }


def test_mapping_mismatch_fails_before_judge_client_is_constructed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = [{"record_id": "r0", "source_id": "source", "source_zh": "诗", "reference_en": "ref"}]
    outputs_path = tmp_path / "outputs.jsonl"
    mapping_path = tmp_path / "mapping.jsonl"
    verification_path = tmp_path / "verification.json"
    write_jsonl(outputs_path, [{"system_id": "s", "record_id": "r0", "status": "ok", "output_text": "translation"}])
    write_jsonl(mapping_path, [{"record_id": "r0", "candidate_map": [{"candidate_id": "A", "system_id": "wrong"}]}])
    constructed = 0

    def fail_constructor():
        nonlocal constructed
        constructed += 1
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(judge, "GPTJudgeClient", fail_constructor)

    with pytest.raises(ValueError, match="mapping"):
        judge.run_judge_evaluation(
            manifest,
            outputs_path=outputs_path,
            raw_path=tmp_path / "raw.jsonl",
            mapping_path=tmp_path / "new_mapping.jsonl",
            system_ids=["s"],
            judge_provider="gpt",
            reuse_mapping_path=mapping_path,
            mapping_verification_path=verification_path,
        )

    assert constructed == 0
    assert json.loads(verification_path.read_text(encoding="utf-8"))["passed"] is False


def test_judge_b_candidate_panels_are_deterministic_and_have_three_or_four_candidates() -> None:
    record = {"record_id": "r0"}
    outputs = {
        (system_id, "r0"): {"status": "ok", "output_text": system_id, "system_name": system_id}
        for system_id in ["d-nt", "d-high", "d-max", "q-nt", "q-2048", "q-4096", "q-default"]
    }

    deepseek_a, _ = judge.build_anonymous_candidates(record, outputs, ["d-nt", "d-high", "d-max"])
    deepseek_b, _ = judge.build_anonymous_candidates(record, outputs, ["d-nt", "d-high", "d-max"])
    qwen, _ = judge.build_anonymous_candidates(record, outputs, ["q-nt", "q-2048", "q-4096", "q-default"])

    assert len(deepseek_a) == 3
    assert deepseek_a == deepseek_b
    assert len(qwen) == 4


def test_manifest_shards_are_disjoint_and_complete() -> None:
    manifest = [{"record_id": f"r{i}"} for i in range(20)]
    shards = [judge.select_manifest_shard(manifest, 3, shard_index) for shard_index in range(3)]
    ids = [{row["record_id"] for row in shard} for shard in shards]

    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == {row["record_id"] for row in manifest}


def test_raw_and_mapping_paths_must_differ_before_reads_force_unlink_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_path = tmp_path / "shared.jsonl"
    shared_path.write_text("sentinel", encoding="utf-8")
    reads = 0
    constructed = 0

    def fail_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("outputs must not be read")

    def fail_constructor():
        nonlocal constructed
        constructed += 1
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(judge, "latest_outputs_by_key", fail_read)
    monkeypatch.setattr(judge, "DeepSeekJudgeClient", fail_constructor)

    with pytest.raises(ValueError, match="different"):
        judge.run_judge_evaluation(
            [],
            outputs_path=tmp_path / "outputs.jsonl",
            raw_path=shared_path,
            mapping_path=shared_path,
            force=True,
        )

    assert shared_path.read_text(encoding="utf-8") == "sentinel"
    assert reads == 0
    assert constructed == 0


@pytest.mark.parametrize(
    ("raw_name", "mapping_name", "no_aggregate"),
    [
        ("shared_raw.jsonl", "shared_mapping.jsonl", True),
        ("judge.shard-0.raw.jsonl", "judge.shard-0.mapping.jsonl", True),
        ("judge.shard-1.raw.jsonl", "judge.shard-1.mapping.jsonl", False),
    ],
)
def test_shard_validation_precedes_force_unlink_and_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_name: str,
    mapping_name: str,
    no_aggregate: bool,
) -> None:
    raw_path = tmp_path / raw_name
    mapping_path = tmp_path / mapping_name
    raw_path.write_text("raw sentinel", encoding="utf-8")
    mapping_path.write_text("mapping sentinel", encoding="utf-8")
    constructed = 0

    def fail_constructor():
        nonlocal constructed
        constructed += 1
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(judge, "DeepSeekJudgeClient", fail_constructor)

    with pytest.raises(ValueError, match="shard"):
        judge.run_judge_evaluation(
            [],
            outputs_path=tmp_path / "outputs.jsonl",
            raw_path=raw_path,
            mapping_path=mapping_path,
            shard_count=2,
            shard_index=1,
            no_aggregate=no_aggregate,
            force=True,
        )

    assert raw_path.read_text(encoding="utf-8") == "raw sentinel"
    assert mapping_path.read_text(encoding="utf-8") == "mapping sentinel"
    assert constructed == 0


def test_hard_gpt_gate_passes_only_complete_translation_and_score_artifacts(tmp_path: Path) -> None:
    count = 2
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    specs = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        write_jsonl(path, translation_rows(system_id, count))
        specs.append((system_id, path))
    write_jsonl(scores_path, score_rows(count))

    report = judge.assert_gpt_stage_gate(dataset_path, specs, scores_path, expected_count=count)

    assert report["complete"] is True
    assert report["translation_gate"]["complete"] is True
    assert report["score_gate"]["complete"] is True


@pytest.mark.parametrize(
    "metric",
    [
        "comet",
        "bertscore_f1",
        "sacrebleu_sentence",
        "chrfpp_sentence",
        "ter_sentence",
        "line_count_diff",
        "length_ratio",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, "1.0", float("nan"), float("inf")])
def test_hard_gpt_score_gate_requires_finite_numeric_metrics(
    tmp_path: Path,
    metric: str,
    invalid_value,
) -> None:
    dataset_path, specs, scores_path = gate_artifacts(tmp_path)
    scores = score_rows(2)
    scores[0][metric] = invalid_value
    write_jsonl(scores_path, scores)

    with pytest.raises(RuntimeError, match="score gate"):
        judge.assert_gpt_stage_gate(dataset_path, specs, scores_path, expected_count=2)


def test_hard_gpt_score_gate_requires_canonical_record_ids(tmp_path: Path) -> None:
    count = 2
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    specs = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        write_jsonl(path, translation_rows(system_id, count))
        specs.append((system_id, path))
    scores = score_rows(count)
    scores[0]["record_id"] = "not-in-canonical-dataset"
    write_jsonl(scores_path, scores)

    with pytest.raises(RuntimeError, match="score gate"):
        judge.assert_gpt_stage_gate(dataset_path, specs, scores_path, expected_count=count)


@pytest.mark.parametrize("missing_field", ["line_count_diff", "length_ratio"])
def test_hard_gpt_score_gate_requires_automatic_diagnostics(
    tmp_path: Path,
    missing_field: str,
) -> None:
    count = 2
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    specs = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        write_jsonl(path, translation_rows(system_id, count))
        specs.append((system_id, path))
    scores = score_rows(count)
    scores[0].pop(missing_field)
    write_jsonl(scores_path, scores)

    with pytest.raises(RuntimeError, match="score gate"):
        judge.assert_gpt_stage_gate(dataset_path, specs, scores_path, expected_count=count)


def test_hard_gpt_gate_binds_each_system_to_its_declared_path(tmp_path: Path) -> None:
    count = 2
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    paths = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        write_jsonl(path, translation_rows(system_id, count))
        paths.append(path)
    write_jsonl(scores_path, score_rows(count))
    mismatched_specs = [
        (GATE_SYSTEMS[0], paths[1]),
        (GATE_SYSTEMS[1], paths[0]),
        (GATE_SYSTEMS[2], paths[2]),
    ]

    with pytest.raises(RuntimeError, match="translation gate"):
        judge.assert_gpt_stage_gate(dataset_path, mismatched_specs, scores_path, expected_count=count)


@pytest.mark.parametrize("failure", ["missing_translation", "duplicate_score", "null_metric"])
def test_failed_hard_gate_prevents_gpt_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    count = 2
    dataset_path = tmp_path / "dataset.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_jsonl(dataset_path, dataset_rows(count))
    specs = []
    for system_id in GATE_SYSTEMS:
        path = tmp_path / f"{system_id}.jsonl"
        rows = translation_rows(system_id, count)
        if failure == "missing_translation" and system_id == GATE_SYSTEMS[0]:
            rows.pop()
        write_jsonl(path, rows)
        specs.append((system_id, path))
    scores = score_rows(count)
    if failure == "duplicate_score":
        scores.append(dict(scores[0]))
    if failure == "null_metric":
        scores[0]["comet"] = None
    write_jsonl(scores_path, scores)
    constructed = 0

    def fail_constructor():
        nonlocal constructed
        constructed += 1
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(judge, "GPTJudgeClient", fail_constructor)

    with pytest.raises(Exception, match="gate"):
        judge.run_judge_evaluation(
            [{"record_id": "r0", "source_id": "source", "source_zh": "诗", "reference_en": "ref"}],
            outputs_path=[path for _, path in specs],
            raw_path=tmp_path / "raw.jsonl",
            mapping_path=tmp_path / "mapping.jsonl",
            system_ids=["s1", "s2"],
            judge_provider="gpt",
            gate_dataset_path=dataset_path,
            gate_translation_specs=specs,
            gate_scores_path=scores_path,
            gate_expected_count=count,
        )

    assert constructed == 0


def test_runner_cli_exposes_gpt_mapping_shard_and_stage_gate_options() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_llm_judge.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in [
        "--judge-provider",
        "--reuse-mapping-path",
        "--mapping-verification-path",
        "--shard-count",
        "--shard-index",
        "--gate-translation",
        "--gate-scores-path",
        "--no-aggregate",
    ]:
        assert option in result.stdout


def test_runner_cli_parses_repeatable_outputs_and_gate_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_llm_judge.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_judge.py",
            "--outputs-path",
            "one.jsonl",
            "--outputs-path",
            "two.jsonl",
            "--gate-translation",
            f"{GATE_SYSTEMS[0]}=deepseek.jsonl",
            "--gate-translation",
            f"{GATE_SYSTEMS[1]}=qwen-low.jsonl",
        ],
    )

    args = runner["parse_args"]()
    gate_specs = runner["parse_gate_translation_specs"](args.gate_translation)

    assert args.outputs_path == [Path("one.jsonl"), Path("two.jsonl")]
    assert gate_specs == [
        (GATE_SYSTEMS[0], Path("deepseek.jsonl")),
        (GATE_SYSTEMS[1], Path("qwen-low.jsonl")),
    ]


def test_shard_no_aggregate_mode_never_writes_scores_or_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_llm_judge.py"))
    globals_ = runner["main"].__globals__
    calls = {"run": 0, "aggregate": 0}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_judge.py",
            "--shard-count",
            "2",
            "--shard-index",
            "1",
            "--raw-path",
            str(tmp_path / "judge.shard-1.raw.jsonl"),
            "--mapping-path",
            str(tmp_path / "judge.shard-1.mapping.jsonl"),
            "--no-aggregate",
        ],
    )
    monkeypatch.setitem(globals_, "load_dotenv", lambda: None)
    monkeypatch.setitem(globals_, "build_judge_manifest", lambda *args, **kwargs: [])
    monkeypatch.setitem(
        globals_,
        "run_judge_evaluation",
        lambda *args, **kwargs: calls.__setitem__("run", calls["run"] + 1),
    )
    monkeypatch.setitem(
        globals_,
        "aggregate_judge_results",
        lambda *args, **kwargs: calls.__setitem__("aggregate", calls["aggregate"] + 1),
    )

    runner["main"]()

    assert calls == {"run": 1, "aggregate": 0}


def test_runner_passes_private_mapping_to_gpt_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_llm_judge.py"))
    globals_ = runner["main"].__globals__
    mapping_path = tmp_path / "private_mapping.jsonl"
    captured: dict = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_judge.py",
            "--judge-provider",
            "gpt",
            "--aggregate-only",
            "--mapping-path",
            str(mapping_path),
        ],
    )
    monkeypatch.setitem(globals_, "load_dotenv", lambda: None)
    monkeypatch.setitem(globals_, "build_judge_manifest", lambda *args, **kwargs: [])

    def fake_aggregate(*args, **kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setitem(globals_, "aggregate_judge_results", fake_aggregate)

    runner["main"]()

    assert captured["private_mapping_path"] == mapping_path


@pytest.mark.parametrize(
    "case",
    [
        "missing_mapping_record",
        "extra_mapping_record",
        "empty_candidate_map",
        "empty_candidate_id",
        "duplicate_candidate_id",
        "empty_system_id",
        "duplicate_system_id",
        "candidate_set_mismatch",
        "ranking_mismatch",
    ],
)
def test_private_mapping_validation_fails_before_scores_or_summary_writes(
    tmp_path: Path,
    case: str,
) -> None:
    raw_rows = [public_raw_row()]
    mapping_rows = [private_mapping_row()]
    if case == "missing_mapping_record":
        mapping_rows = []
    elif case == "extra_mapping_record":
        mapping_rows.append(private_mapping_row("r1"))
    elif case == "empty_candidate_map":
        mapping_rows[0]["candidate_map"] = []
    elif case == "empty_candidate_id":
        mapping_rows[0]["candidate_map"][0]["candidate_id"] = ""
    elif case == "duplicate_candidate_id":
        mapping_rows[0]["candidate_map"][1]["candidate_id"] = "A"
    elif case == "empty_system_id":
        mapping_rows[0]["candidate_map"][0]["system_id"] = ""
    elif case == "duplicate_system_id":
        mapping_rows[0]["candidate_map"][1]["system_id"] = "s1"
    elif case == "candidate_set_mismatch":
        raw_rows[0]["parsed_response"] = valid_judge_payload(["A", "C"])
    elif case == "ranking_mismatch":
        raw_rows[0]["parsed_response"]["ranking"] = ["A", "C"]

    raw_path = tmp_path / "raw.jsonl"
    mapping_path = tmp_path / "mapping.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.csv"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(mapping_path, mapping_rows)
    scores_path.write_text("scores sentinel", encoding="utf-8")
    summary_path.write_text("summary sentinel", encoding="utf-8")

    with pytest.raises(ValueError, match="private mapping"):
        judge.aggregate_judge_results(
            raw_path,
            scores_path,
            summary_path,
            private_mapping_path=mapping_path,
        )

    assert scores_path.read_text(encoding="utf-8") == "scores sentinel"
    assert summary_path.read_text(encoding="utf-8") == "summary sentinel"


def test_legacy_embedded_mapping_aggregation_remains_supported(tmp_path: Path) -> None:
    raw_row = public_raw_row()
    raw_row.update(private_mapping_row())
    raw_path = tmp_path / "raw.jsonl"
    write_jsonl(raw_path, [raw_row])

    scores, _ = judge.aggregate_judge_results(
        raw_path,
        tmp_path / "scores.jsonl",
        tmp_path / "summary.csv",
    )

    assert {row["system_id"] for row in scores} == {"s1", "s2"}


def test_raw_row_records_client_identity_response_model_and_audit_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJudge:
        judge_id = "fake-judge-id"
        model = "requested-model"

        def available(self):
            return True, ""

        def evaluate(self, prompt: str):
            candidate_scores = []
            for candidate_id in ["A", "B"]:
                candidate_scores.append(
                    {
                        "candidate_id": candidate_id,
                        "scores": {dimension: 5.0 for dimension in judge.JUDGE_DIMENSIONS},
                        "reasons": {dimension: "reason" for dimension in judge.JUDGE_DIMENSIONS},
                        "average_score": 5.0,
                        "major_errors": [],
                    }
                )
            parsed = {
                "candidate_scores": candidate_scores,
                "ranking": ["A", "B"],
                "ranking_reason": "reason",
            }
            return {
                "raw_response": {"response": "raw"},
                "content": json.dumps(parsed),
                "reasoning_content": "judge reasoning",
                "token_usage": {"total_tokens": 10},
                "model": "response-model",
                "system_fingerprint": "fp-raw",
                "request_parameters": {"reasoning_effort": "high"},
            }

    outputs_path = tmp_path / "outputs.jsonl"
    write_jsonl(
        outputs_path,
        [
            {"system_id": system_id, "record_id": "r0", "status": "ok", "output_text": system_id}
            for system_id in ["s1", "s2"]
        ],
    )
    raw_path = tmp_path / "raw.jsonl"
    monkeypatch.setattr(judge, "DeepSeekJudgeClient", FakeJudge)

    judge.run_judge_evaluation(
        [{"record_id": "r0", "source_id": "source", "source_zh": "诗", "reference_en": "ref"}],
        outputs_path=outputs_path,
        raw_path=raw_path,
        mapping_path=tmp_path / "mapping.jsonl",
        system_ids=["s1", "s2"],
        sleep_seconds=0,
    )

    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["judge_id"] == "fake-judge-id"
    assert row["judge_model"] == "requested-model"
    assert row["judge_response_model"] == "response-model"
    assert row["judge_request_parameters"] == {"reasoning_effort": "high"}
    assert row["judge_system_fingerprint"] == "fp-raw"
    assert "candidate_map" in row
    assert "missing_candidates" in row


def test_gpt_raw_is_public_safe_and_private_mapping_still_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGPTJudge:
        judge_id = "gpt-test"
        model = "gpt-test-model"

        def available(self):
            return True, ""

        def evaluate(self, prompt: str):
            return {
                "raw_response": {"response": "raw"},
                "content": json.dumps(valid_judge_payload(["A", "B"])),
                "reasoning_content": "judge reasoning",
                "token_usage": {"total_tokens": 10},
                "model": self.model,
                "system_fingerprint": "fp-gpt",
                "request_parameters": {"reasoning_effort": "high"},
            }

    dataset_path, gate_specs, gate_scores_path = gate_artifacts(tmp_path)
    outputs_path = tmp_path / "candidate_outputs.jsonl"
    write_jsonl(
        outputs_path,
        [
            {"system_id": system_id, "record_id": "r0", "status": "ok", "output_text": system_id}
            for system_id in ["s1", "s2"]
        ],
    )
    raw_path = tmp_path / "gpt_raw.jsonl"
    mapping_path = tmp_path / "gpt_mapping.jsonl"
    monkeypatch.setattr(judge, "GPTJudgeClient", FakeGPTJudge)

    judge.run_judge_evaluation(
        [{"record_id": "r0", "source_id": "source", "source_zh": "诗", "reference_en": "ref"}],
        outputs_path=outputs_path,
        raw_path=raw_path,
        mapping_path=mapping_path,
        system_ids=["s1", "s2"],
        judge_provider="gpt",
        gate_dataset_path=dataset_path,
        gate_translation_specs=gate_specs,
        gate_scores_path=gate_scores_path,
        gate_expected_count=2,
        sleep_seconds=0,
    )

    raw_text = raw_path.read_text(encoding="utf-8")
    raw_row = json.loads(raw_text)
    assert "candidate_map" not in raw_row
    assert "missing_candidates" not in raw_row
    assert "s1" not in raw_text and "s2" not in raw_text
    mapping_text = mapping_path.read_text(encoding="utf-8")
    assert "candidate_map" in mapping_text
    assert "s1" in mapping_text and "s2" in mapping_text

    scores, _ = judge.aggregate_judge_results(
        raw_path,
        tmp_path / "scores_out.jsonl",
        tmp_path / "summary.csv",
        private_mapping_path=mapping_path,
    )
    assert len(scores) == 2
    assert {row["system_id"] for row in scores} == {"s1", "s2"}


@pytest.mark.parametrize("mode", ["api_error", "insufficient_candidates"])
def test_gpt_error_rows_do_not_leak_private_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    class FakeGPTJudge:
        judge_id = "gpt-test"
        model = "gpt-test-model"

        def available(self):
            return True, ""

        def evaluate(self, prompt: str):
            raise RuntimeError("simulated judge failure")

    dataset_path, gate_specs, gate_scores_path = gate_artifacts(tmp_path)
    outputs_path = tmp_path / "candidate_outputs.jsonl"
    systems = ["s1"] if mode == "insufficient_candidates" else ["s1", "s2"]
    write_jsonl(
        outputs_path,
        [
            {"system_id": system_id, "record_id": "r0", "status": "ok", "output_text": system_id}
            for system_id in systems
        ],
    )
    raw_path = tmp_path / "gpt_raw.jsonl"
    mapping_path = tmp_path / "gpt_mapping.jsonl"
    monkeypatch.setattr(judge, "GPTJudgeClient", FakeGPTJudge)
    monkeypatch.setenv("JUDGE_MAX_RETRIES", "0")

    judge.run_judge_evaluation(
        [{"record_id": "r0", "source_id": "source", "source_zh": "诗", "reference_en": "ref"}],
        outputs_path=outputs_path,
        raw_path=raw_path,
        mapping_path=mapping_path,
        system_ids=["s1", "s2"],
        judge_provider="gpt",
        gate_dataset_path=dataset_path,
        gate_translation_specs=gate_specs,
        gate_scores_path=gate_scores_path,
        gate_expected_count=2,
        sleep_seconds=0,
    )

    raw_text = raw_path.read_text(encoding="utf-8")
    raw_row = json.loads(raw_text)
    assert raw_row["status"] == "error"
    assert "candidate_map" not in raw_row
    assert "missing_candidates" not in raw_row
    assert "s1" not in raw_text and "s2" not in raw_text
    assert "candidate_map" in mapping_path.read_text(encoding="utf-8")
