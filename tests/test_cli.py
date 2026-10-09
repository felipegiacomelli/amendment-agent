from pydantic import SecretStr

from amendment_agent.cli import prepare, run_config
from amendment_agent.settings import Settings


def test_prepare_builds_a_client_without_sdk_retries() -> None:
    prepared = prepare(
        Settings(_env_file=None, anthropic_api_key=SecretStr("test-key")),  # type: ignore[call-arg]
        None,
    )
    assert isinstance(prepared, tuple)
    client, _ = prepared
    assert client.max_retries == 0
    assert "max_retries" not in run_config(Settings(_env_file=None))  # type: ignore[call-arg]
