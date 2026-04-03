package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

const (
	defaultTopN          = 5
	maxTopN              = 5
	pythonInferenceLimit = 20 * time.Second
)

type RecommendationService struct {
	preferenceRepo     *repos.PreferenceRepo
	rentalRepo         *repos.RentalRepo
	recommendationRepo *repos.RecommendationRepo
	pythonExecutable   string
	pythonScript       string
}

type recommendationRequest struct {
	UserPreferences  models.Preference `json:"user_preferences"`
	CandidateListings []models.Rental  `json:"candidate_listings"`
	TopN             int               `json:"top_n"`
}

type recommendationResponse struct {
	Recommendations []models.RecommendedRental `json:"recommendations"`
	Rows            []models.UserRecommendation `json:"rows"`
	Error           string                      `json:"error"`
	Details         string                      `json:"details"`
}

func NewRecommendationService(
	preferenceRepo *repos.PreferenceRepo,
	rentalRepo *repos.RentalRepo,
	recommendationRepo *repos.RecommendationRepo,
) *RecommendationService {
	pythonExecutable := strings.TrimSpace(os.Getenv("PYTHON_EXECUTABLE"))
	if pythonExecutable == "" {
		pythonExecutable = "python"
	}

	return &RecommendationService{
		preferenceRepo:     preferenceRepo,
		rentalRepo:         rentalRepo,
		recommendationRepo: recommendationRepo,
		pythonExecutable:   pythonExecutable,
		pythonScript:       resolvePythonScriptPath(),
	}
}

func resolvePythonScriptPath() string {
	if configuredPath := strings.TrimSpace(os.Getenv("PY_RECOMMENDER_SCRIPT")); configuredPath != "" {
		return configuredPath
	}

	pathInBackend := filepath.Join("py", "services", "recommend_cli.py")
	if _, err := os.Stat(pathInBackend); err == nil {
		return pathInBackend
	}

	pathFromWorkspace := filepath.Join("Backend", "py", "services", "recommend_cli.py")
	if _, err := os.Stat(pathFromWorkspace); err == nil {
		return pathFromWorkspace
	}

	return pathInBackend
}

func (s *RecommendationService) GetByUserID(userID int, topN int, persist bool) ([]models.RecommendedRental, error) {
	if topN <= 0 {
		topN = defaultTopN
	}
	if topN > maxTopN {
		topN = maxTopN
	}

	preference, err := s.preferenceRepo.GetByUserID(userID)
	if err != nil {
		return []models.RecommendedRental{}, nil
	}

	allListings, err := s.rentalRepo.GetAll()
	if err != nil {
		return nil, err
	}

	candidateListings := make([]models.Rental, 0, len(allListings))
	for _, listing := range allListings {
		if listing.UserID == userID {
			continue
		}
		candidateListings = append(candidateListings, listing)
	}

	if len(candidateListings) == 0 {
		return []models.RecommendedRental{}, nil
	}

	recommendations, rows, err := s.inferRecommendations(preference, candidateListings, topN)
	if err != nil {
		return nil, err
	}

	if persist {
		rowsToPersist := rows
		if len(rowsToPersist) == 0 {
			rowsToPersist = make([]models.UserRecommendation, 0, len(recommendations))
			for _, rec := range recommendations {
				rowsToPersist = append(rowsToPersist, models.UserRecommendation{
					UserID:    userID,
					ListingID: rec.ListingID,
					Score:     rec.RecommendationScore,
				})
			}
		}

		for i := range rowsToPersist {
			rowsToPersist[i].UserID = userID
		}

		if err := s.recommendationRepo.ReplaceForUser(userID, rowsToPersist); err != nil {
			return nil, err
		}
	}

	return recommendations, nil
}

func (s *RecommendationService) inferRecommendations(
	preference models.Preference,
	candidateListings []models.Rental,
	topN int,
) ([]models.RecommendedRental, []models.UserRecommendation, error) {
	payload := recommendationRequest{
		UserPreferences:  preference,
		CandidateListings: candidateListings,
		TopN:             topN,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, nil, fmt.Errorf("infer recommendations: marshal payload: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), pythonInferenceLimit)
	defer cancel()

	cmd := exec.CommandContext(ctx, s.pythonExecutable, s.pythonScript)
	cmd.Stdin = bytes.NewReader(payloadBytes)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, nil, fmt.Errorf("infer recommendations: python process timeout")
		}

		stderrText := strings.TrimSpace(stderr.String())
		if stderrText != "" {
			return nil, nil, fmt.Errorf("infer recommendations: %v: %s", err, stderrText)
		}

		return nil, nil, fmt.Errorf("infer recommendations: %v", err)
	}

	var response recommendationResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return nil, nil, fmt.Errorf("infer recommendations: parse python response: %v", err)
	}

	if response.Error != "" {
		if response.Details != "" {
			return nil, nil, fmt.Errorf("infer recommendations: %s (%s)", response.Error, response.Details)
		}
		return nil, nil, fmt.Errorf("infer recommendations: %s", response.Error)
	}

	return response.Recommendations, response.Rows, nil
}