package repos

import (
	"database/sql"
	"fmt"
)

type AuthRepo struct {
	db *sql.DB
}

func NewAuthRepo(db *sql.DB) *AuthRepo {
    return &AuthRepo{
		db: db,
    }
}

func (r *AuthRepo) Signup(e [32]byte, p [32]byte) error {
	_, err := r.db.Exec("INSERT INTO user (email, password) VALUES (?, ?)", e, p)
	if err != nil {
		return fmt.Errorf("Signup: %v", err)
	}

	return nil
}

func (r *AuthRepo) Login(e [32]byte, p [32]byte) ([32]byte, error) {
	var p2 [32]byte

	row := r.db.QueryRow("SELECT password FROM users WHERE email = ?", e)
	if err := row.Scan(&p2); err != nil {
		if err == sql.ErrNoRows {
			return p2, fmt.Errorf("Login %q: no such email", e)
		}
		return p2, fmt.Errorf("Login %q: %v", e, err)
	}

	return p2, nil
}

