package routes

import (
	"github.com/gofiber/fiber/v2"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func RecommendationRoutes(router fiber.Router, recommendationHandler *handlers.RecommendationHandler) {
	recommendationGroup := router.Group("/recommendations")

	recommendationGroup.Get("/", recommendationHandler.GetByUserID)
}