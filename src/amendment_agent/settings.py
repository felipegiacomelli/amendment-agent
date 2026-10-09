"""Settings read from the environment and `.env`, plus the fixed repo paths."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Paths are relative to the repo root (the CLI runs from there). They are constants,
# not settings: they never change. Functions take directories as parameters so tests
# can pass tmp_path instead.
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
TEXT_DIR = DATA_DIR / "text"
MANIFEST_PATH = DATA_DIR / "manifest.yaml"
OUTPUT_DIR = Path("outputs")
RUNS_DIR = Path("runs")
EVALS_DIR = Path("evals")
CASES_PATH = EVALS_DIR / "cases.jsonl"
RESULTS_DIR = EVALS_DIR / "results"


class Settings(BaseSettings):
    """Each command checks only the fields it needs (see cli.py)."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    anthropic_api_key: SecretStr | None = None
    sec_user_agent: str | None = None
    agent_model: str = "claude-sonnet-5-5"
    agent_max_turns: int = 25
    agent_max_cost_usd: float = 1.00
    agent_request_timeout_s: float = 120.0
