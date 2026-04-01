package handlers

import (
	"strconv"

	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type PreferenceHandler struct {
	validate *validator.Validate
	service  *services.PreferenceService
}

func NewPreferenceHandler(v *validator.Validate, s *services.PreferenceService) *PreferenceHandler {
	return &PreferenceHandler{validate: v, service: s}
}

func (h *PreferenceHandler) GetByUserID(c *fiber.Ctx) error {
	userIDParam := c.Query("user_id")
	if userIDParam == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id is required"})
	}

	userID, err := strconv.Atoi(userIDParam)
	if err != nil || userID <= 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id must be a positive integer"})
	}

	pref, getErr := h.service.GetByUserID(userID)
	if getErr != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": getErr.Error()})
	}

	return c.Status(fiber.StatusOK).JSON(pref)
}

func (h *PreferenceHandler) Upsert(c *fiber.Ctx) error {
	var pref models.Preference

	if err := c.BodyParser(&pref); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	if err := h.validate.Struct(pref); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	if pref.BudgetMin != nil && pref.BudgetMax != nil && *pref.BudgetMin > *pref.BudgetMax {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "budget_min cannot be greater than budget_max"})
	}

	if err := h.service.Upsert(pref); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{"message": "Preferences saved successfully"})
}
