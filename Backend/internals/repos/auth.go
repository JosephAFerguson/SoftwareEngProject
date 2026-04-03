package repos

import (
	"database/sql"
	"errors"
	"fmt"

	"github.com/go-sql-driver/mysql"
)

type AuthRepo struct {
	db *sql.DB
}

func NewAuthRepo(db *sql.DB) *AuthRepo {
	return &AuthRepo{db: db}
}

func (r *AuthRepo) Signup(email string, passwordHash string) (int, error) {
	result, err := r.db.Exec(
		"INSERT INTO users (email, password) VALUES (?, ?)",
		email,
		passwordHash,
	)
	if err != nil {
		var mysqlErr *mysql.MySQLError
		if errors.As(err, &mysqlErr) && mysqlErr.Number == 1062 {
			return 0, ErrDuplicateEmail
		}

		return 0, fmt.Errorf("signup: %w", err)
	}

	insertedID, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("signup: failed to read inserted user id: %w", err)
	}

	return int(insertedID), nil
}

func (r *AuthRepo) Login(email string) (int, string, error) {
	var userID int
	var passwordHash string

	row := r.db.QueryRow(
		"SELECT user_id, password FROM users WHERE email = ?",
		email,
	)
	if err := row.Scan(&userID, &passwordHash); err != nil {
		if err == sql.ErrNoRows {
			return 0, "", fmt.Errorf("login %q: no such email", email)
		}
		return 0, "", fmt.Errorf("login %q: %w", email, err)
	}

	return userID, passwordHash, nil
}