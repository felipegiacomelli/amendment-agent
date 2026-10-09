from pytest import MonkeyPatch

from amendment_agent.settings import Settings


def test_empty_env_values_count_as_missing(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MAX_TURNS", "")
    monkeypatch.setenv("AGENT_MODEL", "")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.agent_max_turns == 25
    assert settings.agent_model == "claude-sonnet-5-5"
