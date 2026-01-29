package routes

import (
    "github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func AdminRoutes(router fiber.Router, adminHandler *handlers.AdminHandler) {
    adminGroup := router.Group("/admin")

    adminGroup.Post("/health", adminHandler.Health)
}

