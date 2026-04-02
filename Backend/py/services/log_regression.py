from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


Record = Mapping[str, Any] | Any


PREFERENCE_COLUMNS = (
	"user_id",
	"preferred_location",
	"budget_min",
	"budget_max",
	"preferred_roommates",
	"preferred_bednum",
	"preferred_bathnum",
)

LISTING_COLUMNS = (
	"listing_id",
	"user_id",
	"title",
	"address",
	"price",
	"sqft",
	"roommates",
	"bednum",
	"bathnum",
	"pet_friendly",
	"available_from",
	"available_to",
)

USER_RECOMMENDATION_COLUMNS = (
	"user_id",
	"listing_id",
	"score",
)


def _get(record: Record, key: str, default: Any = None) -> Any:
	if isinstance(record, Mapping):
		return record.get(key, default)
	return getattr(record, key, default)


def _as_float(value: Any, default: float | None = None) -> float | None:
	if value is None or value == "":
		return default
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _as_int(value: Any, default: int | None = None) -> int | None:
	if value is None or value == "":
		return default
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


def _as_bool(value: Any, default: bool = False) -> bool:
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		normalized = value.strip().lower()
		if normalized in {"1", "true", "t", "yes", "y"}:
			return True
		if normalized in {"0", "false", "f", "no", "n"}:
			return False
	return default


def _normalized_text(value: Any) -> str:
	return str(value or "").strip().lower()


def _parse_date(value: Any) -> date | None:
	if value is None or value == "":
		return None

	if isinstance(value, datetime):
		return value.date()

	if isinstance(value, date):
		return value

	if isinstance(value, str):
		try:
			return date.fromisoformat(value)
		except ValueError:
			return None

	return None


def _location_match(preferred_location: Any, listing: Record) -> float:
	preferred = _normalized_text(preferred_location)
	if not preferred:
		return 0.0

	tokens = [token for token in preferred.replace(",", " ").split() if token]
	if not tokens:
		return 0.0

	listing_text = " ".join(
		_normalized_text(_get(listing, field, ""))
		for field in ("title", "address")
	)

	return 1.0 if any(token in listing_text for token in tokens) else 0.0


def _bounded_similarity(preferred: float | None, actual: float | None, tolerance: float) -> float:
	if preferred is None or actual is None:
		return 0.0
	if tolerance <= 0:
		return 0.0
	return max(0.0, 1.0 - (abs(actual - preferred) / tolerance))


def _budget_features(preference: Record, listing: Record) -> tuple[float, float]:
	price = _as_float(_get(listing, "price"), 0.0) or 0.0
	budget_min = _as_float(_get(preference, "budget_min"), None)
	budget_max = _as_float(_get(preference, "budget_max"), None)

	if budget_min is None and budget_max is None:
		return 0.0, 0.0

	if budget_min is None:
		budget_min = 0.0

	if budget_max is None:
		budget_max = budget_min

	if budget_max < budget_min:
		budget_min, budget_max = budget_max, budget_min

	within_budget = 1.0 if budget_min <= price <= budget_max else 0.0
	budget_span = max(budget_max - budget_min, 1.0)
	budget_center = (budget_min + budget_max) / 2.0
	alignment = max(0.0, 1.0 - (abs(price - budget_center) / budget_span))

	return within_budget, alignment


def _availability_feature(listing: Record) -> float:
	start = _parse_date(_get(listing, "available_from"))
	end = _parse_date(_get(listing, "available_to"))
	today = date.today()

	if start is None and end is None:
		return 0.0

	if start is not None and today < start:
		return 0.0

	if end is not None and today > end:
		return 0.0

	return 1.0


def _feature_vector(preference: Record, listing: Record) -> list[float]:
	within_budget, budget_alignment = _budget_features(preference, listing)

	preferred_bednum = _as_int(_get(preference, "preferred_bednum"), None)
	preferred_bathnum = _as_float(_get(preference, "preferred_bathnum"), None)
	preferred_roommates = _as_int(_get(preference, "preferred_roommates"), None)

	listing_bednum = _as_int(_get(listing, "bednum"), None)
	listing_bathnum = _as_float(_get(listing, "bathnum"), None)
	listing_roommates = _as_int(_get(listing, "roommates"), None)

	bed_match = _bounded_similarity(preferred_bednum, listing_bednum, tolerance=3.0)
	bath_match = _bounded_similarity(preferred_bathnum, listing_bathnum, tolerance=2.0)
	roommate_match = _bounded_similarity(preferred_roommates, listing_roommates, tolerance=4.0)
	location_match = _location_match(_get(preference, "preferred_location"), listing)
	pet_friendly = 1.0 if _as_bool(_get(listing, "pet_friendly"), False) else 0.0
	sqft = _as_float(_get(listing, "sqft"), 0.0) or 0.0
	sqft_score = min(sqft / 2000.0, 1.0)
	availability_score = _availability_feature(listing)

	return [
		within_budget,
		budget_alignment,
		bed_match,
		bath_match,
		roommate_match,
		location_match,
		pet_friendly,
		sqft_score,
		availability_score,
	]


def _heuristic_label(preference: Record, listing: Record) -> int:
	features = _feature_vector(preference, listing)
	heuristic_score = (
		2.5 * features[0]
		+ 1.25 * features[1]
		+ 1.5 * features[2]
		+ 1.0 * features[3]
		+ 0.75 * features[4]
		+ 2.0 * features[5]
		+ 0.5 * features[6]
		+ 0.25 * features[7]
		+ 0.75 * features[8]
	)
	return 1 if heuristic_score >= 4.0 else 0


SAMPLE_PREFERENCES: list[dict[str, Any]] = [
	{
		"user_id": 1,
		"preferred_location": "Boston",
		"budget_min": 900,
		"budget_max": 1600,
		"preferred_roommates": 1,
		"preferred_bednum": 1,
		"preferred_bathnum": 1.0,
	},
	{
		"user_id": 2,
		"preferred_location": "Cambridge",
		"budget_min": 1200,
		"budget_max": 2100,
		"preferred_roommates": 2,
		"preferred_bednum": 2,
		"preferred_bathnum": 1.0,
	},
	{
		"user_id": 3,
		"preferred_location": "Seattle",
		"budget_min": 1800,
		"budget_max": 2700,
		"preferred_roommates": 1,
		"preferred_bednum": 2,
		"preferred_bathnum": 2.0,
	},
	{
		"user_id": 4,
		"preferred_location": "Austin",
		"budget_min": 1000,
		"budget_max": 1900,
		"preferred_roommates": 2,
		"preferred_bednum": 3,
		"preferred_bathnum": 2.0,
	},
	{
		"user_id": 5,
		"preferred_location": "Somerville",
		"budget_min": 750,
		"budget_max": 1350,
		"preferred_roommates": 0,
		"preferred_bednum": 1,
		"preferred_bathnum": 1.0,
	},
]


SAMPLE_LISTINGS: list[dict[str, Any]] = [
	{
		"listing_id": 101,
		"user_id": 9,
		"title": "Bright Boston Studio",
		"address": "Boston, MA",
		"price": 1350,
		"sqft": 520,
		"roommates": 0,
		"bednum": 1,
		"bathnum": 1.0,
		"pet_friendly": True,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 102,
		"user_id": 10,
		"title": "Cambridge Two Bedroom",
		"address": "Cambridge, MA",
		"price": 1890,
		"sqft": 820,
		"roommates": 2,
		"bednum": 2,
		"bathnum": 1.0,
		"pet_friendly": False,
		"available_from": "2026-01-01",
		"available_to": "2026-12-31",
	},
	{
		"listing_id": 103,
		"user_id": 11,
		"title": "Somerville Cozy Flat",
		"address": "Somerville, MA",
		"price": 1120,
		"sqft": 610,
		"roommates": 0,
		"bednum": 1,
		"bathnum": 1.0,
		"pet_friendly": True,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 104,
		"user_id": 12,
		"title": "Seattle Modern Two Bed",
		"address": "Seattle, WA",
		"price": 2360,
		"sqft": 910,
		"roommates": 1,
		"bednum": 2,
		"bathnum": 2.0,
		"pet_friendly": True,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 105,
		"user_id": 13,
		"title": "Austin Family Home",
		"address": "Austin, TX",
		"price": 1720,
		"sqft": 1420,
		"roommates": 2,
		"bednum": 3,
		"bathnum": 2.0,
		"pet_friendly": True,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 106,
		"user_id": 14,
		"title": "Chicago Budget Room",
		"address": "Chicago, IL",
		"price": 950,
		"sqft": 430,
		"roommates": 1,
		"bednum": 1,
		"bathnum": 1.0,
		"pet_friendly": False,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 107,
		"user_id": 15,
		"title": "Downtown Cambridge Luxury Loft",
		"address": "Cambridge, MA",
		"price": 1990,
		"sqft": 920,
		"roommates": 2,
		"bednum": 2,
		"bathnum": 1.5,
		"pet_friendly": True,
		"available_from": "2026-01-01",
		"available_to": None,
	},
	{
		"listing_id": 108,
		"user_id": 16,
		"title": "Quiet Boston One Bed",
		"address": "Boston, MA",
		"price": 1450,
		"sqft": 650,
		"roommates": 1,
		"bednum": 1,
		"bathnum": 1.0,
		"pet_friendly": False,
		"available_from": "2026-01-01",
		"available_to": None,
	},
]


def _build_heuristic_training_data(
	preferences: Sequence[Record],
	listings: Sequence[Record],
) -> tuple[list[list[float]], list[int]]:
	features: list[list[float]] = []
	labels: list[int] = []

	for preference in preferences:
		for listing in listings:
			features.append(_feature_vector(preference, listing))
			labels.append(_heuristic_label(preference, listing))

	return features, labels


def build_training_data_from_recommendation_rows(
	preferences: Sequence[Record],
	listings: Sequence[Record],
	recommendation_rows: Sequence[Record],
	positive_threshold: float = 0.6,
) -> tuple[list[list[float]], list[int]]:
	"""Build supervised training data from user_recommendations rows.

	Rows with score >= positive_threshold are treated as class 1; otherwise class 0.
	Records that cannot be joined by user_id/listing_id are ignored.
	"""

	preferences_by_user: dict[int, Record] = {}
	for preference in preferences:
		user_id = _as_int(_get(preference, "user_id"), None)
		if user_id is not None:
			preferences_by_user[user_id] = preference

	listings_by_id: dict[int, Record] = {}
	for listing in listings:
		listing_id = _as_int(_get(listing, "listing_id"), None)
		if listing_id is not None:
			listings_by_id[listing_id] = listing

	features: list[list[float]] = []
	labels: list[int] = []

	for row in recommendation_rows:
		user_id = _as_int(_get(row, "user_id"), None)
		listing_id = _as_int(_get(row, "listing_id"), None)
		score = _as_float(_get(row, "score"), None)

		if user_id is None or listing_id is None or score is None:
			continue

		preference = preferences_by_user.get(user_id)
		listing = listings_by_id.get(listing_id)

		if preference is None or listing is None:
			continue

		features.append(_feature_vector(preference, listing))
		labels.append(1 if score >= positive_threshold else 0)

	return features, labels


def train_logistic_regression_model(
	preferences: Sequence[Record] | None = None,
	listings: Sequence[Record] | None = None,
	recommendation_rows: Sequence[Record] | None = None,
	positive_threshold: float = 0.6,
) -> Pipeline:
	"""Train a lightweight logistic-regression ranking model.

	If recommendation_rows are provided, they are used as the primary supervised
	signal. If they are missing or insufficient, heuristic fallback data is used.
	"""

	training_preferences = list(preferences or SAMPLE_PREFERENCES)
	training_listings = list(listings or SAMPLE_LISTINGS)

	features: list[list[float]] = []
	labels: list[int] = []

	if recommendation_rows:
		features, labels = build_training_data_from_recommendation_rows(
			training_preferences,
			training_listings,
			recommendation_rows,
			positive_threshold=positive_threshold,
		)

	if len(features) < 20 or len(set(labels)) < 2:
		fallback_features, fallback_labels = _build_heuristic_training_data(
			training_preferences,
			training_listings,
		)
		features.extend(fallback_features)
		labels.extend(fallback_labels)

	if len(features) == 0:
		raise ValueError("Training data is empty")

	if len(set(labels)) < 2:
		raise ValueError("Training data must include both positive and negative examples")

	model = Pipeline(
		steps=[
			("scaler", StandardScaler()),
			(
				"classifier",
				LogisticRegression(
					max_iter=1000,
					solver="liblinear",
					class_weight="balanced",
					random_state=42,
				),
			),
		],
	)
	model.fit(features, labels)
	return model


@lru_cache(maxsize=1)
def get_default_model() -> Pipeline:
	"""Return a cached default model trained on  sample data."""

	return train_logistic_regression_model()


DEFAULT_MODEL = get_default_model()


def score_listing(
	user_preferences: Record,
	listing: Record,
	model: Pipeline | None = None,
) -> float:
	"""Return P(relevant=1) for a user/listing pair."""

	active_model = model or DEFAULT_MODEL
	proba = active_model.predict_proba([_feature_vector(user_preferences, listing)])[0][1]
	return float(proba)


def recommend_listings(
	user_preferences: Record,
	candidate_listings: Sequence[Record],
	top_n: int = 5,
	model: Pipeline | None = None,
) -> list[dict[str, Any]]:
	"""Return top recommendations, capped to 5."""

	if not candidate_listings:
		return []

	if top_n <= 0:
		return []

	bounded_top_n = min(int(top_n), 5)
	active_model = model or DEFAULT_MODEL
	scored: list[dict[str, Any]] = []

	for listing in candidate_listings:
		listing_id = _as_int(_get(listing, "listing_id"), None)
		if listing_id is None:
			continue

		score = round(score_listing(user_preferences, listing, active_model), 6)

		if isinstance(listing, Mapping):
			listing_payload = dict(listing)
		else:
			listing_payload = {
				"listing_id": listing_id,
				"user_id": _as_int(_get(listing, "user_id"), None),
				"title": _get(listing, "title", ""),
				"address": _get(listing, "address", ""),
				"price": _as_int(_get(listing, "price"), None),
				"sqft": _as_int(_get(listing, "sqft"), None),
				"roommates": _as_int(_get(listing, "roommates"), None),
				"bednum": _as_int(_get(listing, "bednum"), None),
				"bathnum": _as_float(_get(listing, "bathnum"), None),
				"pet_friendly": _as_bool(_get(listing, "pet_friendly"), False),
				"available_from": _get(listing, "available_from", None),
				"available_to": _get(listing, "available_to", None),
			}

		listing_payload["recommendation_score"] = score
		scored.append(listing_payload)

	scored.sort(
		key=lambda item: (
			item["recommendation_score"],
			_as_int(item.get("listing_id"), 0) or 0,
		),
		reverse=True,
	)

	return scored[:bounded_top_n]


def to_user_recommendation_rows(
	user_id: int,
	recommendations: Sequence[Record],
) -> list[dict[str, Any]]:
	"""Convert ranked results into rows matching user_recommendations schema."""

	rows: list[dict[str, Any]] = []

	for rec in recommendations:
		listing_id = _as_int(_get(rec, "listing_id"), None)
		score = _as_float(_get(rec, "recommendation_score"), None)
		if listing_id is None or score is None:
			continue

		rows.append(
			{
				"user_id": int(user_id),
				"listing_id": int(listing_id),
				"score": float(score),
			}
		)

	return rows


if __name__ == "__main__":
	sample_pref = SAMPLE_PREFERENCES[0]
	recs = recommend_listings(sample_pref, SAMPLE_LISTINGS, top_n=5)
	rows = to_user_recommendation_rows(sample_pref["user_id"], recs)
	print(recs)
	print(rows)