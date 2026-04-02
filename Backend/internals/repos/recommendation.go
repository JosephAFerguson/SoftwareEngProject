package repos

import (
	"database/sql"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
)

type RecommendationRepo struct {
	db *sql.DB
}

func NewRecommendationRepo(db *sql.DB) *RecommendationRepo {
	return &RecommendationRepo{db: db}
}

func (r *RecommendationRepo) ReplaceForUser(userID int, recommendations []models.UserRecommendation) error {
	tx, err := r.db.Begin()
	if err != nil {
		return fmt.Errorf("ReplaceForUser %d: begin tx: %v", userID, err)
	}

	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	_, err = tx.Exec(`DELETE FROM user_recommendations WHERE user_id = ?`, userID)
	if err != nil {
		return fmt.Errorf("ReplaceForUser %d: delete existing rows: %v", userID, err)
	}

	for _, rec := range recommendations {
		_, err = tx.Exec(
			`INSERT INTO user_recommendations (user_id, listing_id, score) VALUES (?, ?, ?)`,
			userID,
			rec.ListingID,
			rec.Score,
		)
		if err != nil {
			return fmt.Errorf("ReplaceForUser %d: insert listing %d: %v", userID, rec.ListingID, err)
		}
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("ReplaceForUser %d: commit tx: %v", userID, err)
	}

	return nil
}