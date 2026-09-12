"""Credential and safety-rail loading.

Everything the client needs comes from the environment so that no secret is
ever a literal in source. Missing credentials fail loudly at startup rather
than producing a confusing 401 later.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when the environment is not usable for trading."""


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    api_key: str
    private_key: str
    dry_run: bool
    max_order_usd: float

    @classmethod
    def load(cls) -> "Config":
        api_key = os.getenv("RH_API_KEY", "").strip()
        private_key = os.getenv("RH_PRIVATE_KEY", "").strip()

        missing = [
            name
            for name, value in (("RH_API_KEY", api_key), ("RH_PRIVATE_KEY", private_key))
            if not value
        ]
        if missing:
            raise ConfigError(
                f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in."
            )

        try:
            max_order_usd = float(os.getenv("RH_MAX_ORDER_USD", "10"))
        except ValueError as exc:
            raise ConfigError("RH_MAX_ORDER_USD must be a number.") from exc

        if max_order_usd <= 0:
            raise ConfigError("RH_MAX_ORDER_USD must be greater than zero.")

        # Default to dry run. An unset or malformed value must never mean "live".
        return cls(
            api_key=api_key,
            private_key=private_key,
            dry_run=_flag("RH_DRY_RUN", "true"),
            max_order_usd=max_order_usd,
        )
