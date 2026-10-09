"""Tests for the layered configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from yonga.config.settings import DevicePolicy, Settings, load_settings


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_match_the_contract(tmp_path: Path) -> None:
    settings = load_settings(config_file=tmp_path / "missing.toml", env={})

    assert settings.default_device is DevicePolicy.AUTO
    assert settings.offline is False
    assert settings.download_confirm_threshold_bytes == 5 * 2**30
    assert settings.temp_min_bytes == 20 * 2**30
    assert settings.temp_size_multiplier == pytest.approx(2.5)
    assert settings.temp_safety_margin_bytes == 5 * 2**30
    assert settings.compile_cache_enabled is True
    assert settings.show_reasoning is False
    assert settings.hash_weights is True
    assert settings.bench_runs == 3
    assert settings.bench_warmups == 1
    assert settings.max_new_tokens == 4096
    assert settings.registry_update_policy == "manual"
    assert settings.telemetry is False
    assert settings.json_output is False
    assert settings.config_warnings == []


def test_toml_top_level_and_table_keys(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        bench_runs = 7

        [yonga]
        default_device = "gpu"
        offline = true
        """,
    )
    settings = load_settings(config_file=config, env={})

    assert settings.bench_runs == 7
    assert settings.default_device is DevicePolicy.GPU
    assert settings.offline is True
    assert settings.config_file == config


def test_env_beats_config_file(tmp_path: Path) -> None:
    config = write_config(tmp_path, 'default_device = "gpu"\nbench_runs = 7\n')
    settings = load_settings(
        config_file=config,
        env={"YONGA_DEFAULT_DEVICE": "npu"},
    )

    assert settings.default_device is DevicePolicy.NPU
    assert settings.bench_runs == 7


def test_overrides_beat_env_and_config_file(tmp_path: Path) -> None:
    config = write_config(tmp_path, 'default_device = "gpu"\n')
    settings = load_settings(
        config_file=config,
        env={"YONGA_DEFAULT_DEVICE": "npu", "YONGA_BENCH_RUNS": "9"},
        overrides={"default_device": "cpu"},
    )

    assert settings.default_device is DevicePolicy.CPU
    assert settings.bench_runs == 9


def test_none_valued_overrides_are_ignored(tmp_path: Path) -> None:
    config = write_config(tmp_path, 'default_device = "gpu"\n')
    settings = load_settings(
        config_file=config,
        env={},
        overrides={"default_device": None, "verbose": True},
    )

    assert settings.default_device is DevicePolicy.GPU
    assert settings.verbose is True


def test_full_precedence_chain(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        [yonga]
        max_new_tokens = 111
        bench_runs = 2
        offline = true
        """,
    )
    settings = load_settings(
        config_file=config,
        env={"YONGA_MAX_NEW_TOKENS": "222", "YONGA_BENCH_RUNS": "5"},
        overrides={"max_new_tokens": 333},
    )

    assert settings.max_new_tokens == 333  # override wins
    assert settings.bench_runs == 5  # env wins
    assert settings.offline is True  # config file wins over the default
    assert settings.show_reasoning is False  # default


def test_unknown_keys_become_warnings_not_crashes(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        bench_runs = 4
        nonsense_key = 1

        [yonga]
        other_nonsense = "x"
        """,
    )
    settings = load_settings(
        config_file=config,
        env={"YONGA_MYSTERY": "1"},
        overrides={"also_unknown": 2},
    )

    assert settings.bench_runs == 4
    joined = " ".join(settings.config_warnings)
    assert "nonsense_key" in joined
    assert "other_nonsense" in joined
    assert "YONGA_MYSTERY" in joined
    assert "also_unknown" in joined


def test_secret_env_variables_do_not_warn(tmp_path: Path) -> None:
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"YONGA_HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz0123456789"},
    )

    assert settings.config_warnings == []


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_telemetry_true_is_rejected_from_env(tmp_path: Path, value: str) -> None:
    with pytest.raises(Exception, match=r"(?i)telemetry") as excinfo:
        load_settings(config_file=tmp_path / "missing.toml", env={"YONGA_TELEMETRY": value})

    assert type(excinfo.value).__name__.endswith("Error")


def test_telemetry_true_is_rejected_from_config_file(tmp_path: Path) -> None:
    config = write_config(tmp_path, "telemetry = true\n")

    with pytest.raises(Exception, match=r"(?i)telemetry"):
        load_settings(config_file=config, env={})


def test_telemetry_true_is_rejected_from_overrides(tmp_path: Path) -> None:
    with pytest.raises(Exception, match=r"(?i)telemetry"):
        load_settings(
            config_file=tmp_path / "missing.toml",
            env={},
            overrides={"telemetry": True},
        )


def test_telemetry_false_is_accepted(tmp_path: Path) -> None:
    config = write_config(tmp_path, "telemetry = false\n")
    settings = load_settings(config_file=config, env={})

    assert settings.telemetry is False


def test_direct_construction_also_rejects_telemetry() -> None:
    with pytest.raises(Exception, match=r"(?i)telemetry"):
        Settings(telemetry=True)


def test_invalid_value_raises_a_configuration_error(tmp_path: Path) -> None:
    config = write_config(tmp_path, "bench_runs = 0\n")

    with pytest.raises(Exception, match="bench_runs"):
        load_settings(config_file=config, env={})


def test_invalid_toml_raises_a_configuration_error(tmp_path: Path) -> None:
    config = write_config(tmp_path, "this is not = = toml\n")

    with pytest.raises(Exception, match=r"(?i)toml"):
        load_settings(config_file=config, env={})


def test_device_policy_casing_is_normalised(tmp_path: Path) -> None:
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"YONGA_DEFAULT_DEVICE": " NPU "},
    )

    assert settings.default_device is DevicePolicy.NPU


def test_paths_follow_the_configured_environment(tmp_path: Path) -> None:
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path / "share")},
    )

    assert settings.paths.data_dir == tmp_path / "share" / "yonga"
    assert settings.paths.cache_dir == tmp_path / ".cache" / "yonga"


def test_explicit_directories_override_the_environment(tmp_path: Path) -> None:
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"HOME": str(tmp_path)},
        overrides={"data_dir": tmp_path / "custom"},
    )

    assert settings.paths.data_dir == tmp_path / "custom"


def test_a_relative_data_dir_override_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"HOME": str(tmp_path)},
        overrides={"data_dir": Path("relative") / "data"},
    )

    assert settings.paths.data_dir == tmp_path / "relative" / "data"


def test_config_dir_override_moves_the_default_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text("bench_runs = 6\n", encoding="utf-8")

    settings = load_settings(
        env={"HOME": str(tmp_path), "YONGA_CONFIG_DIR": str(config_dir)},
    )

    assert settings.bench_runs == 6
    assert settings.config_file == config_dir / "config.toml"


def test_missing_default_config_file_is_not_an_error(tmp_path: Path) -> None:
    settings = load_settings(env={"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "c")})

    assert settings.config_file is None
    assert settings.config_warnings == []


def test_debug_implies_verbose(tmp_path: Path) -> None:
    settings = load_settings(config_file=tmp_path / "missing.toml", env={"YONGA_DEBUG": "true"})

    assert settings.debug is True
    assert settings.verbose is True


def test_redacted_dump_hides_secrets(tmp_path: Path) -> None:
    secret = "hf_abcdefghijklmnopqrstuvwxyz0123456789"
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"YONGA_HF_HOME": f"/tmp/cache-{secret}"},
    )
    dump = settings.redacted_dump()

    assert secret not in repr(dump)
    assert dump["max_new_tokens"] == 4096
    assert dump["telemetry"] is False


def test_paths_property_is_usable(tmp_path: Path) -> None:
    settings = load_settings(
        config_file=tmp_path / "missing.toml",
        env={"HOME": str(tmp_path)},
    )
    settings.paths.ensure()

    assert settings.paths.logs_dir.is_dir()
