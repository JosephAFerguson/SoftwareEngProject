package repos

import (
	"database/sql"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
)

type ProfileRepo struct {
	db *sql.DB
}

func NewProfileRepo(db *sql.DB) *ProfileRepo {
	return &ProfileRepo{db: db}
}

func (r *ProfileRepo) GetByUserID(userID int) (models.UserProfile, error) {
	var profile models.UserProfile
	profile.UserID = userID

	row := r.db.QueryRow(`SELECT name, profile_photo FROM users WHERE user_id = ?`, userID)
	if err := row.Scan(&profile.Name, &profile.ProfilePhoto); err != nil {
		if err == sql.ErrNoRows {
			return profile, fmt.Errorf("Get Profile for user %d: not found", userID)
		}
		return profile, fmt.Errorf("Get Profile for user %d: %v", userID, err)
	}

	return profile, nil
}

func (r *ProfileRepo) Update(profile models.UserProfile) error {
	result, err := r.db.Exec(`UPDATE users SET name = ?, profile_photo = ? WHERE user_id = ?`,
		profile.Name,
		profile.ProfilePhoto,
		profile.UserID,
	)
	if err != nil {
		return fmt.Errorf("Update Profile for user %d: %v", profile.UserID, err)
	}

	affected, affErr := result.RowsAffected()
	if affErr != nil {
		return fmt.Errorf("Update Profile for user %d: %v", profile.UserID, affErr)
	}
	if affected == 0 {
		return fmt.Errorf("Update Profile for user %d: not found", profile.UserID)
	}

	return nil
}
