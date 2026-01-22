package main

import (
	"log"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/routes"
	"github.com/gofiber/fiber/v2"
)

const PORT = "3000"

func main() {
	app := fiber.New()

	api := app.Group("/api")

	v1 := api.Group("/v1", func(c *fiber.Ctx) error {
		c.Set("Version", "v1")
		return c.Next()
	})

	routes.AdminRoutes(v1)
	routes.AuthRoutes(v1)
	routes.UserRoutes(v1)

	log.Fatal(app.Listen(":" + PORT))
}

