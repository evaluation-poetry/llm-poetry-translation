from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "supplemental"


def _load(name: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _walk(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_supplemental_config_set_is_complete_and_public_safe() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.json")} == {
        "generation.json",
        "judges.json",
        "scoring.json",
        "statistical_tests.json",
    }

    forbidden_keys = re.compile(
        r"(?:api.?key|credential|password|secret|endpoint|base.?url|"
        r"(?:^|_)path$|(?:^|_)(?:sha\d*|hash)(?:$|_)|mapping|account.?id)",
        re.IGNORECASE,
    )
    absolute_path = re.compile(r"^(?:[A-Za-z]:[\\/]|/home/|/Users/|/mnt/)")

    for config_path in CONFIG_DIR.glob("*.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in _walk(payload):
            assert not forbidden_keys.search(key), (config_path.name, key)
            if isinstance(value, str):
                assert not value.startswith(("http://", "https://")), (config_path.name, value)
                assert not absolute_path.search(value), (config_path.name, value)
                assert "codex" not in value.lower()
                assert "claude" not in value.lower()


def test_generation_config_records_frozen_request_parameters() -> None:
    config = _load("generation.json")
    systems = {row["system_id"]: row for row in config["systems"]}

    assert config["corpus_records"] == 397
    assert config["prompt_version"] == "understand_translate_v1"
    assert config["input_policy"] == {
        "source_zh_only": True,
        "reference_translation_sent": False,
        "system_identity_sent": False,
    }
    assert set(systems) == {
        "deepseek_v4_flash_thinking",
        "deepseek_v4_flash_thinking_max",
        "qwen36_plus_thinking",
        "qwen36_plus_thinking_b2048",
        "qwen36_plus_thinking_b4096",
        "qwen35_9b_open_non_thinking",
        "qwen35_9b_open_thinking",
    }

    assert systems["deepseek_v4_flash_thinking_max"]["request"] == {
        "max_tokens": 8192,
        "reasoning_effort": "max",
        "stream": False,
        "thinking": "enabled",
    }
    assert systems["qwen36_plus_thinking_b2048"]["request"]["thinking_budget"] == 2048
    assert systems["qwen36_plus_thinking_b4096"]["request"]["thinking_budget"] == 4096
    assert systems["qwen36_plus_thinking"]["request"]["thinking_budget_sent"] is False

    for system_id in ("qwen35_9b_open_non_thinking", "qwen35_9b_open_thinking"):
        assert systems[system_id]["model"] == "Qwen/Qwen3.5-9B"
        assert systems[system_id]["request"]["max_tokens"] == 16384
        assert systems[system_id]["request"]["seed"] == 20260716
    assert systems["qwen35_9b_open_non_thinking"]["request"]["enable_thinking"] is False
    assert systems["qwen35_9b_open_thinking"]["request"]["enable_thinking"] is True

    runtime = config["open_weight_runtime"]
    assert runtime["dtype"] == "bfloat16"
    assert runtime["vllm_version"] == "0.25.1"
    assert runtime["max_model_len"] == 32768
    assert runtime["max_num_seqs"] == 24
    assert runtime["speculative_decoding"] == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    assert config["documented_exception"] == {
        "affected_outputs": 1,
        "system_id": "qwen35_9b_open_thinking",
        "retry_max_tokens": 30000,
        "other_request_parameters_unchanged": True,
    }


def test_scoring_and_judge_configs_match_implementations() -> None:
    scoring = _load("scoring.json")
    assert scoring["score_input"] == "final_translation_only"
    assert scoring["reasoning_content_scored"] is False
    assert scoring["comet"] == {
        "model": "Unbabel/wmt22-comet-da",
        "batch_size": 4,
        "gpus": 0,
    }
    assert scoring["bertscore"] == {
        "component": "F1",
        "language": "en",
        "model": "microsoft/deberta-xlarge-mnli",
        "batch_size": 64,
    }
    assert scoring["sacrebleu"]["chrf_word_order"] == 2
    assert scoring["sacrebleu"]["ter_segmentation"] == "paired_nonempty_poem_lines"

    judges = _load("judges.json")
    assert judges["prompt_version"] == "agents_md_judge_v1_20260514"
    assert judges["sample_seed"] == 20260513
    assert judges["score_increment"] == 0.5
    assert judges["dimensions"] == 11
    by_id = {row["judge_id"]: row for row in judges["runs"]}
    assert by_id["gpt55_high_judge_a"]["reasoning_effort"] == "high"
    assert by_id["gpt55_high_judge_b"]["reasoning_effort"] == "high"
    assert by_id["deepseek_v4_pro_non_thinking"]["thinking"] == "disabled"
    assert by_id["deepseek_v4_pro_non_thinking"]["temperature"] == 0
    assert by_id["deepseek_v4_pro_non_thinking"]["temperature_sent"] is True
    assert by_id["gpt55_high_judge_a"]["temperature_sent"] is False
    assert by_id["gpt55_high_judge_b"]["temperature_sent"] is False
    assert all(row["top_p_sent"] is False for row in judges["runs"])


def test_statistical_config_names_every_published_entry_point() -> None:
    config = _load("statistical_tests.json")
    scripts = {row["script"]: row for row in config["analyses"]}
    assert set(scripts) == {
        "scripts/stats_alignment.py",
        "scripts/stats_crossjudge.py",
        "scripts/stats_human_paired.py",
        "scripts/stats_intensity.py",
        "scripts/stats_judge_paired.py",
        "scripts/stats_open_qwen35.py",
        "scripts/stats_rebuttal_auxiliary.py",
    }
    assert scripts["scripts/stats_crossjudge.py"]["bootstrap_samples"] == 10000
    assert scripts["scripts/stats_crossjudge.py"]["permutation_samples"] == 10000
    assert scripts["scripts/stats_crossjudge.py"]["minimum_valid_fraction"] == 0.8
    assert scripts["scripts/stats_open_qwen35.py"]["seed"] == 20260716
    assert scripts["scripts/stats_intensity.py"]["seed"] == 20260706
    assert config["multiple_testing"] == "Holm family-wise correction"


def test_documented_prompt_versions_exist() -> None:
    prompt_text = (ROOT / "docs" / "prompts.md").read_text(encoding="utf-8")
    for prompt_version in (
        "understand_translate_v1",
        "understand_translate_v2_final_only",
        "direct_v1",
        "agents_md_judge_v1_20260514",
    ):
        assert f"`{prompt_version}`" in prompt_text
