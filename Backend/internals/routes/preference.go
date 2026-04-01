package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func PreferenceRoutes(router fiber.Router, preferenceHandler *handlers.PreferenceHandler) {
	preferencesGroup := router.Group("/preferences")

	preferencesGroup.Get("/", preferenceHandler.GetByUserID)
	preferencesGroup.Put("/", preferenceHandler.Upsert)
}
