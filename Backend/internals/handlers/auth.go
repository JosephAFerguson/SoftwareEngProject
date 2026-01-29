package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type AuthHandler struct {
    validate *validator.Validate
	service *services.AuthService
}

func NewAuthHandler(v *validator.Validate, s *services.AuthService) *AuthHandler {
    return &AuthHandler{
        validate: v,
		service: s,
    }
}

// (TODO) write better error handling
func (h *AuthHandler) Signup(c *fiber.Ctx) error {
	var user models.User

	if err := c.BodyParser(&user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}

	if err := h.validate.Struct(user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}

	if err := h.service.Signup(user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}

	return c.SendStatus(fiber.StatusOK)
}

func (h *AuthHandler) Login(c *fiber.Ctx) error {
	var user models.User

	if err := c.BodyParser(&user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}

	if err := h.validate.Struct(user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}

	if err := h.service.Login(user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(err)
	}


	return c.SendStatus(fiber.StatusOK)
}

