package repos

import (
	"database/sql"
	"fmt"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
)

type PreferenceRepo struct {
	db *sql.DB
}

func NewPreferenceRepo(db *sql.DB) *PreferenceRepo {
	return &PreferenceRepo{db: db}
}

func (r *PreferenceRepo) GetByUserID(userID int) (models.Preference, error) {
	var p models.Preference

	row := r.db.QueryRow(`SELECT preference_id, user_id, preferred_location, budget_min, budget_max, preferred_roommates, preferred_bednum, preferred_bathnum
		FROM preferences WHERE user_id = ? ORDER BY preference_id DESC LIMIT 1`, userID)

	if err := row.Scan(
		&p.PreferenceID,
		&p.UserID,
		&p.PreferredLocation,
		&p.BudgetMin,
		&p.BudgetMax,
		&p.PreferredRoommates,
		&p.PreferredBednum,
		&p.PreferredBathnum,
	); err != nil {
		if err == sql.ErrNoRows {
			return p, fmt.Errorf("Get Preference for user %d: not found", userID)
		}
		return p, fmt.Errorf("Get Preference for user %d: %v", userID, err)
	}

	return p, nil
}

func (r *PreferenceRepo) Upsert(pref models.Preference) error {
	var existingPreferenceID int
	err := r.db.QueryRow(`SELECT preference_id FROM preferences WHERE user_id = ? ORDER BY preference_id DESC LIMIT 1`, pref.UserID).Scan(&existingPreferenceID)

	if err == sql.ErrNoRows {
		_, insertErr := r.db.Exec(`INSERT INTO preferences (user_id, preferred_location, budget_min, budget_max, preferred_roommates, preferred_bednum, preferred_bathnum)
			VALUES (?, ?, ?, ?, ?, ?, ?)`,
			pref.UserID,
			pref.PreferredLocation,
			pref.BudgetMin,
			pref.BudgetMax,
			pref.PreferredRoommates,
			pref.PreferredBednum,
			pref.PreferredBathnum,
		)
		if insertErr != nil {
			return fmt.Errorf("Insert Preference for user %d: %v", pref.UserID, insertErr)
		}
		return nil
	}

	if err != nil {
		return fmt.Errorf("Upsert Preference for user %d: %v", pref.UserID, err)
	}

	_, updateErr := r.db.Exec(`UPDATE preferences
		SET preferred_location = ?, budget_min = ?, budget_max = ?, preferred_roommates = ?, preferred_bednum = ?, preferred_bathnum = ?
		WHERE preference_id = ?`,
		pref.PreferredLocation,
		pref.BudgetMin,
		pref.BudgetMax,
		pref.PreferredRoommates,
		pref.PreferredBednum,
		pref.PreferredBathnum,
		existingPreferenceID,
	)
	if updateErr != nil {
		return fmt.Errorf("Update Preference for user %d: %v", pref.UserID, updateErr)
	}

	return nil
}
