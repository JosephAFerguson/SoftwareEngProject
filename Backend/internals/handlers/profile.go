package handlers

import (
	"strconv"

	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type ProfileHandler struct {
	validate *validator.Validate
	service  *services.ProfileService
}

func NewProfileHandler(v *validator.Validate, s *services.ProfileService) *ProfileHandler {
	return &ProfileHandler{validate: v, service: s}
}

func (h *ProfileHandler) GetByUserID(c *fiber.Ctx) error {
	userIDParam := c.Query("user_id")
	if userIDParam == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id is required"})
	}

	userID, err := strconv.Atoi(userIDParam)
	if err != nil || userID <= 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id must be a positive integer"})
	}

	profile, getErr := h.service.GetByUserID(userID)
	if getErr != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": getErr.Error()})
	}

	return c.Status(fiber.StatusOK).JSON(profile)
}

func (h *ProfileHandler) Update(c *fiber.Ctx) error {
	var profile models.UserProfile

	if err := c.BodyParser(&profile); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	if err := h.validate.Struct(profile); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	if err := h.service.Update(profile); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{"message": "Profile updated successfully"})
}
