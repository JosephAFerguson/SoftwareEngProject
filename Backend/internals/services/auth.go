package services

import (
	"crypto/sha256"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

type AuthService struct {
	repo *repos.AuthRepo
}

func NewAuthService(r *repos.AuthRepo) *AuthService {
    return &AuthService{
		repo: r,
    }
}

func (s *AuthService) Signup(user models.User) error {
	// encrypts data
	e := sha256.Sum256([]byte(user.Email))
	p := sha256.Sum256([]byte(user.Password))

	err := s.repo.Signup(e, p)

	return err
}

func (s *AuthService) Login(user models.User) error {
	// encrypts data
	e := sha256.Sum256([]byte(user.Email))
	p := sha256.Sum256([]byte(user.Password))

	p2, err := s.repo.Login(e, p)
	if p2 != p {
		return fmt.Errorf("Incorrect Password")
	}

	return err
}

