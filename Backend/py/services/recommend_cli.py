from __future__ import annotations

import json
import sys
from typing import Any

from log_regression import recommend_listings, to_user_recommendation_rows


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")))


def _error(error: str, details: str) -> int:
    _emit({"error": error, "details": details})
    return 1


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return _error("missing_input", "Request body is empty")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error("invalid_json", str(exc))

    user_preferences = payload.get("user_preferences")
    candidate_listings = payload.get("candidate_listings") or []
    top_n_raw = payload.get("top_n", 5)

    if not isinstance(user_preferences, dict):
        return _error("invalid_user_preferences", "user_preferences must be a JSON object")

    if not isinstance(candidate_listings, list):
        return _error("invalid_candidate_listings", "candidate_listings must be a JSON array")

    try:
        top_n = int(top_n_raw)
    except (TypeError, ValueError):
        top_n = 5

    top_n = max(1, min(top_n, 5))

    try:
        recommendations = recommend_listings(user_preferences, candidate_listings, top_n=top_n)
    except Exception as exc:  # noqa: BLE001 - surface model errors to API caller
        return _error("inference_failed", str(exc))

    user_id_raw = user_preferences.get("user_id")
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        user_id = 0

    rows = to_user_recommendation_rows(user_id, recommendations) if user_id > 0 else []

    _emit(
        {
            "recommendations": recommendations,
            "rows": rows,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())