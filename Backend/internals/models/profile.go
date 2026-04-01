package models

type UserProfile struct {
	UserID       int     `json:"user_id" validate:"required,gt=0"`
	Name         *string `json:"name" validate:"omitempty,max=100"`
	ProfilePhoto *string `json:"profile_photo" validate:"omitempty,max=255"`
}
