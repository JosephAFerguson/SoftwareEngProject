package controllers

import (
	"github.com/gofiber/fiber/v2"
)

func GetUserInfo(c *fiber.Ctx) error {
	//Check DB for user info

	return c.SendStatus(fiber.StatusOK)
}

