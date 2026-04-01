package services

import (
	"errors"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

var ErrUserAlreadyHosting = errors.New("user already has a hosting")

type RentalService struct {
	repo *repos.RentalRepo
}

func NewRentalService(r *repos.RentalRepo) *RentalService {
    return &RentalService{
		repo: r,
    }
}

func (s *RentalService) Post(rental models.Rental) error {
	hasListing, err := s.repo.UserHasListing(rental.UserID)
	if err != nil {
		return err
	}

	if hasListing {
		return fmt.Errorf("%w: user_id=%d", ErrUserAlreadyHosting, rental.UserID)
	}

	err = s.repo.Post(rental)

	return err
}

func (s *RentalService) Get(address string) (models.Rental, error) {
	re, err := s.repo.Get(address)

	return re, err
}

func (s *RentalService) GetAll() ([]models.Rental, error) {
	rentals, err := s.repo.GetAll()

	return rentals, err
}

