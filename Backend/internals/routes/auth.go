package routes

import (
	"github.com/gofiber/fiber/v2"
	//"github.com/JosephAFerguson/SoftwareEngProject/internals/controllers"
)

func AuthRoutes(router fiber.Router) {
	// Add logic and DB connections

	authGroup := router.Group("/auth") 

	// temporary handlers for these routes
	authGroup.Get("/signup", func (c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusOK)
	})

	authGroup.Get("/login/:user", func (c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusOK)
	})
}

