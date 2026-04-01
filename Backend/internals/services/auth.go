package services

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

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

func (s *AuthService) Signup(user models.User) (int, error) {
	email := strings.TrimSpace(strings.ToLower(user.Email))
	p := sha256.Sum256([]byte(user.Password))
	passwordHash := hex.EncodeToString(p[:])

	userID, err := s.repo.Signup(email, passwordHash)

	return userID, err
}

func (s *AuthService) Login(user models.User) (int, error) {
	email := strings.TrimSpace(strings.ToLower(user.Email))
	p := sha256.Sum256([]byte(user.Password))
	passwordHash := hex.EncodeToString(p[:])

	userID, p2, err := s.repo.Login(email)
	if err != nil {
		return 0, err
	}

	if p2 != passwordHash {
		return 0, fmt.Errorf("Incorrect Password")
	}

	return userID, nil
}

