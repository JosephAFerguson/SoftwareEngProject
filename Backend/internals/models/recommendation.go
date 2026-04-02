package models

type UserRecommendation struct {
	UserID    int     `json:"user_id"`
	ListingID int     `json:"listing_id"`
	Score     float64 `json:"score"`
}

type RecommendedRental struct {
	ListingID            int      `json:"listing_id"`
	UserID               int      `json:"user_id"`
	Address              string   `json:"address"`
	Title                string   `json:"title"`
	Price                int      `json:"price"`
	Sqft                 *int     `json:"sqft"`
	Roommates            *int     `json:"roommates"`
	Bednum               *int     `json:"bednum"`
	Bathnum              *float64 `json:"bathnum"`
	PetFriendly          bool     `json:"pet_friendly"`
	AvailableFrom        *Date    `json:"available_from"`
	AvailableTo          *Date    `json:"available_to"`
	Photos               []string `json:"photos"`
	RecommendationScore  float64  `json:"recommendation_score"`
}