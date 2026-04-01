package models

import (
	"database/sql/driver"
	"fmt"
	"time"
)

// Date represents a date without time, for JSON marshaling/unmarshaling
type Date struct {
	time.Time
}

// UnmarshalJSON parses a date string in YYYY-MM-DD format
func (d *Date) UnmarshalJSON(data []byte) error {
	// Remove quotes from JSON string
	str := string(data)
	if len(str) >= 2 && str[0] == '"' && str[len(str)-1] == '"' {
		str = str[1 : len(str)-1]
	}

	// Parse date in YYYY-MM-DD format
	if str == "" {
		d.Time = time.Time{}
		return nil
	}

	parsed, err := time.Parse("2006-01-02", str)
	if err != nil {
		return err
	}

	d.Time = parsed
	return nil
}

// MarshalJSON formats the date as YYYY-MM-DD
func (d Date) MarshalJSON() ([]byte, error) {
	if d.Time.IsZero() {
		return []byte("null"), nil
	}
	return []byte(`"` + d.Time.Format("2006-01-02") + `"`), nil
}

// Scan implements sql.Scanner for database scanning
func (d *Date) Scan(value interface{}) error {
	if value == nil {
		d.Time = time.Time{}
		return nil
	}

	switch v := value.(type) {
	case time.Time:
		d.Time = v
	case []byte:
		parsed, err := time.Parse("2006-01-02", string(v))
		if err != nil {
			return err
		}
		d.Time = parsed
	case string:
		if v == "" {
			d.Time = time.Time{}
			return nil
		}
		parsed, err := time.Parse("2006-01-02", v)
		if err != nil {
			return err
		}
		d.Time = parsed
	default:
		return fmt.Errorf("cannot scan %T into Date", value)
	}

	return nil
}

// Value implements driver.Valuer for database storage
func (d Date) Value() (driver.Value, error) {
	if d.Time.IsZero() {
		return nil, nil
	}
	return d.Time.Format("2006-01-02"), nil
}

type Rental struct {
	ListingID     int      `json:"listing_id"`
	UserID        int      `json:"user_id" validate:"required"`
	Address       string   `json:"address" validate:"required,max=255"`
	Title         string   `json:"title" validate:"required,max=255"`
	Price         int      `json:"price" validate:"required,gt=0"`
	Sqft          *int     `json:"sqft" validate:"omitempty,gt=0"`
	Roommates     *int     `json:"roommates" validate:"omitempty,gte=0"`
	Bednum        *int     `json:"bednum" validate:"omitempty,gte=0"`
	Bathnum       *float64 `json:"bathnum" validate:"omitempty,gte=0"`
	PetFriendly   bool     `json:"pet_friendly"`
	AvailableFrom *Date    `json:"available_from"`
	AvailableTo   *Date    `json:"available_to"`
	Photos        []string `json:"photos"`
}

