import torch

# Load model ONCE
model = torch.load("model/recommender.pt")
model.eval()

def recommend_listings(user_preferences: dict):
    """
    user_preferences example:
    {
      "location": "Boston",
      "price_max": 1200,
      "bedrooms": 2
    }
    """

    # Example: convert inputs to tensor (simplified)
    features = torch.tensor([
        user_preferences.get("price_max", 0),
        user_preferences.get("bedrooms", 0)
    ], dtype=torch.float32)

    with torch.no_grad():
        scores = model(features)

    # Dummy return for now
    return {
        "recommended_listing_ids": [3, 7, 12]
    }
