"""Compatibility wrapper for the logistic-regression recommender."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:
	from .log_regression import DEFAULT_MODEL, recommend_listings as _recommend_listings
except ImportError:
	from log_regression import DEFAULT_MODEL, recommend_listings as _recommend_listings


model = DEFAULT_MODEL


def recommend_listings(
	user_preferences: Mapping[str, Any],
	candidate_listings: Sequence[Mapping[str, Any]] | None = None,
	top_n: int = 5,
) -> list[dict[str, Any]]:
	"""Rank listings with the shared logistic-regression model.

	The caller should pass the available listings to score. Returning a list of
	ranked listings makes it easy to render 3-5 recommendations in the UI.
	"""

	if candidate_listings is None:
		return []

	return _recommend_listings(user_preferences, candidate_listings, top_n=top_n, model=model)
