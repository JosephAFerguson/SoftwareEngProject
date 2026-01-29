package middleware

import (
	"github.com/gofiber/fiber/v2"
	//"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
)

func UserExistsMiddleware(c *fiber.Ctx) error {
	userID := c.Params("user")

	if userID == "" {
		return fiber.ErrBadRequest
	}

	//Check DB for user

	c.Locals("user", userID)
	return c.Next()
}

