package routes

import "github.com/gofiber/fiber/v2"

func AdminRoutes(router fiber.Router) {
	//Eventually add auth for Admin accts

    router.Get("/health", func(c *fiber.Ctx) error {
        return c.SendString("ok")
    })
}

