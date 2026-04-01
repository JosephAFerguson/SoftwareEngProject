package repos

import (
	"database/sql"
	"encoding/json"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
)

type RentalRepo struct {
	db *sql.DB
}

func NewRentalRepo(db *sql.DB) *RentalRepo {
    return &RentalRepo{
		db: db,
    }
}

func (r *RentalRepo) Post(rental models.Rental) error {
	// Marshal photos to JSON
	var photosJSON []byte
	if rental.Photos != nil {
		var err error
		photosJSON, err = json.Marshal(rental.Photos)
		if err != nil {
			return fmt.Errorf("Post rental: failed to marshal photos: %v", err)
		}
	}

	_, err := r.db.Exec(`INSERT INTO listings (user_id, title, address, price, sqft, roommates, bednum, bathnum, pet_friendly, available_from, available_to, photos) 
						VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
						rental.UserID, rental.Title, rental.Address, rental.Price, rental.Sqft, rental.Roommates,
						rental.Bednum, rental.Bathnum, rental.PetFriendly, rental.AvailableFrom, rental.AvailableTo, photosJSON)
	if err != nil {
		return fmt.Errorf("Post rental: %v", err)
	}

	return nil
}

func (r *RentalRepo) UserHasListing(userID int) (bool, error) {
	var count int

	err := r.db.QueryRow(`SELECT COUNT(1) FROM listings WHERE user_id = ?`, userID).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("UserHasListing %d: %v", userID, err)
	}

	return count > 0, nil
}

func (r *RentalRepo) Get(a string) (models.Rental, error) {
	var re models.Rental
	var photosJSON []byte

	row := r.db.QueryRow(`SELECT listing_id, user_id, title, address, price, sqft, roommates, bednum, bathnum, pet_friendly, available_from, available_to, photos 
						 FROM listings WHERE address = ?`, a)
	if err := row.Scan(
		&re.ListingID, &re.UserID, &re.Title, &re.Address, &re.Price, &re.Sqft, &re.Roommates,
		&re.Bednum, &re.Bathnum, &re.PetFriendly, &re.AvailableFrom, &re.AvailableTo, &photosJSON); err != nil {
		if err == sql.ErrNoRows {
			return re, fmt.Errorf("Get Rental %q: no such address", a)
		}
		return re, fmt.Errorf("Get Rental %q: %v", a, err)
	}

	// Parse photos JSON
	if photosJSON != nil {
		if err := json.Unmarshal(photosJSON, &re.Photos); err != nil {
			return re, fmt.Errorf("Get Rental %q: failed to parse photos: %v", a, err)
		}
	}

	return re, nil
}

func (r *RentalRepo) GetAll() ([]models.Rental, error) {
	rows, err := r.db.Query(`SELECT listing_id, user_id, title, address, price, sqft, roommates, bednum, bathnum, pet_friendly, available_from, available_to, photos 
							FROM listings`)
	if err != nil {
		return nil, fmt.Errorf("GetAll Rentals: %v", err)
	}
	defer rows.Close()

	var rentals []models.Rental
	for rows.Next() {
		var re models.Rental
		var photosJSON []byte
		
		if err := rows.Scan(
			&re.ListingID, &re.UserID, &re.Title, &re.Address, &re.Price, &re.Sqft, &re.Roommates,
			&re.Bednum, &re.Bathnum, &re.PetFriendly, &re.AvailableFrom, &re.AvailableTo, &photosJSON); err != nil {
			return nil, fmt.Errorf("GetAll Rentals scan: %v", err)
		}

		// Parse photos JSON
		if photosJSON != nil {
			if err := json.Unmarshal(photosJSON, &re.Photos); err != nil {
				return nil, fmt.Errorf("GetAll Rentals: failed to parse photos: %v", err)
			}
		}

		rentals = append(rentals, re)
	}

	return rentals, nil
}

