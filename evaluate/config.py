from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_TEST_EMBEDDINGS = ROOT_DIR / "data" / "voxceleb_test_embeddings.pt"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "results" / "speaker_search_evaluation.json"
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_LIMIT = 25
DEFAULT_METRICS = ("cosine",)


def load_environment() -> dict[str, str]:
    from dotenv import load_dotenv

    dotenv_path = ROOT_DIR / "data_base" / ".env"
    if not dotenv_path.exists():
        raise FileNotFoundError(
            f"Could not find .env file at {dotenv_path}. Make sure the script is run from data_base/."
        )

    load_dotenv(dotenv_path)

    env = {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
    }

    missing = [key for key, value in env.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required DB environment variables in {dotenv_path}: {', '.join(missing)}"
        )

    return {key: value for key, value in env.items() if value is not None}


def build_connection_string(env: dict[str, str]) -> str:
    return (
        f"host={env['DB_HOST']} port={env['DB_PORT']} dbname={env['DB_NAME']} "
        f"user={env['DB_USER']} password={env['DB_PASSWORD']}"
    )
