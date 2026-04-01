package routes

import (
    "github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
)

func RentalRoutes(router fiber.Router, rentalHandler *handlers.RentalHandler) {
    rentalGroup := router.Group("/rental")

    rentalGroup.Post("/", rentalHandler.Post)
	rentalGroup.Get("/", rentalHandler.Get)
	rentalGroup.Get("/all", rentalHandler.GetAll)
}

