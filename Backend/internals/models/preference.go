package models

type Preference struct {
	PreferenceID        int      `json:"preference_id"`
	UserID              int      `json:"user_id" validate:"required,gt=0"`
	PreferredLocation   *string  `json:"preferred_location" validate:"omitempty,max=255"`
	BudgetMin           *int     `json:"budget_min" validate:"omitempty,gte=0"`
	BudgetMax           *int     `json:"budget_max" validate:"omitempty,gte=0"`
	PreferredRoommates  *int     `json:"preferred_roommates" validate:"omitempty,gte=0"`
	PreferredBednum     *int     `json:"preferred_bednum" validate:"omitempty,gte=0"`
	PreferredBathnum    *float64 `json:"preferred_bathnum" validate:"omitempty,gte=0"`
}
