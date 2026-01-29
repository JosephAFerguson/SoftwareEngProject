package routes

import (
    "github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func AuthRoutes(router fiber.Router, authHandler *handlers.AuthHandler) {
    authGroup := router.Group("/auth")

    authGroup.Post("/signup", authHandler.Signup)
	authGroup.Post("/login", authHandler.Login)
}

