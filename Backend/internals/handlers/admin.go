package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	//"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	//"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type AdminHandler struct {
    validate *validator.Validate
	//service *services.AuthService (maybe needed later)
}

func NewAdminHandler(v *validator.Validate) *AdminHandler {
    return &AdminHandler{
        validate: v,
		//service: s,
    }
}

func (h *AdminHandler) Health(c *fiber.Ctx) error {
	return c.SendStatus(fiber.StatusOK)
}

