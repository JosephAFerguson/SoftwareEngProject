package models

type Rental struct {
	Address string `json:"address" validate:"required,max=75"`
	Price *int `json:"price" validate:"required,gt=0"`
	Sqft  *int `json:"sqft" validate:"omitempty,gt=0"`
	Roommates  *int `json:"roommates" validate:"omitempty,gte=0"`
	Bednum  *int `json:"bednum" validate:"omitempty,gte=0"`
	Bathnum  *int `json:"bathnum" validate:"omitempty,gte=0"`
}

