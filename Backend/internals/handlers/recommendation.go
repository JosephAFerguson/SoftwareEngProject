package handlers

import (
	"strconv"

	"github.com/gofiber/fiber/v2"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type RecommendationHandler struct {
	service *services.RecommendationService
}

func NewRecommendationHandler(s *services.RecommendationService) *RecommendationHandler {
	return &RecommendationHandler{service: s}
}

func (h *RecommendationHandler) GetByUserID(c *fiber.Ctx) error {
	userIDParam := c.Query("user_id")
	if userIDParam == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id is required"})
	}

	userID, err := strconv.Atoi(userIDParam)
	if err != nil || userID <= 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "user_id must be a positive integer"})
	}

	topN := 5
	if topNParam := c.Query("top_n"); topNParam != "" {
		parsedTopN, parseErr := strconv.Atoi(topNParam)
		if parseErr != nil || parsedTopN <= 0 || parsedTopN > 5 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "top_n must be an integer between 1 and 5"})
		}
		topN = parsedTopN
	}

	persist := false
	if persistParam := c.Query("persist"); persistParam != "" {
		parsedPersist, parseErr := strconv.ParseBool(persistParam)
		if parseErr != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "persist must be true or false"})
		}
		persist = parsedPersist
	}

	recommendations, getErr := h.service.GetByUserID(userID, topN, persist)
	if getErr != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": getErr.Error()})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"recommendations": recommendations,
		"count":           len(recommendations),
		"persisted":       persist,
	})
}