package services

import (
	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

type PreferenceService struct {
	repo *repos.PreferenceRepo
}

func NewPreferenceService(r *repos.PreferenceRepo) *PreferenceService {
	return &PreferenceService{repo: r}
}

func (s *PreferenceService) GetByUserID(userID int) (models.Preference, error) {
	return s.repo.GetByUserID(userID)
}

func (s *PreferenceService) Upsert(pref models.Preference) error {
	return s.repo.Upsert(pref)
}
