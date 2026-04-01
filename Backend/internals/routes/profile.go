package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func ProfileRoutes(router fiber.Router, profileHandler *handlers.ProfileHandler) {
	profileGroup := router.Group("/profile")

	profileGroup.Get("/", profileHandler.GetByUserID)
	profileGroup.Put("/", profileHandler.Update)
}
