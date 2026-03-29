package repos

import (
	"database/sql"
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

func (r *RentalRepo) Post(a string, p *int, s *int, 
						  re *int, be *int, ba *int) error {
	_, err := r.db.Exec("INSERT INTO rental VALUES (?, ?, ?, ?, ?, ?)", 
						a, &p, &s, &re, &be, &ba)
	if err != nil {
		return fmt.Errorf("Signup: %v", err)
	}

	return nil
}

func (r *RentalRepo) Get(a string) (models.Rental, error) {
	var re models.Rental 

	row := r.db.QueryRow("SELECT * FROM rental WHERE address = ?", a)
	if err := row.Scan(
		&re.Address, &re.Price, &re.Sqft, &re.Roommates,
		&re.Bednum, &re.Bathnum); err != nil {
		if err == sql.ErrNoRows {
			return re, fmt.Errorf("Get Rental %q: no such address", a)
		}
		return re, fmt.Errorf("Get Rental %q: %v", a, err)
	}

	return re, nil
}

