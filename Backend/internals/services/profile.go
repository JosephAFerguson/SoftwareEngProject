package services

import (
	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

type ProfileService struct {
	repo *repos.ProfileRepo
}

func NewProfileService(r *repos.ProfileRepo) *ProfileService {
	return &ProfileService{repo: r}
}

func (s *ProfileService) GetByUserID(userID int) (models.UserProfile, error) {
	return s.repo.GetByUserID(userID)
}

func (s *ProfileService) Update(profile models.UserProfile) error {
	return s.repo.Update(profile)
}
