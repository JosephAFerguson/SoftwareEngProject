package services

import (
	//"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

type RentalService struct {
	repo *repos.RentalRepo
}

func NewRentalService(r *repos.RentalRepo) *RentalService {
    return &RentalService{
		repo: r,
    }
}

func (s *RentalService) Post(rental models.Rental) error {
	err := s.repo.Post(
		rental.Address, rental.Price, rental.Sqft,
		rental.Roommates, rental.Bednum, rental.Bathnum,
	)

	return err
}

func (s *RentalService) Get(address string) (models.Rental, error) {
	re, err := s.repo.Get(address)

	return re, err
}

