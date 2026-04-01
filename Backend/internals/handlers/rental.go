package handlers

import (
	"errors"

	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type RentalHandler struct {
    validate *validator.Validate
	service *services.RentalService
}

func NewRentalHandler(v *validator.Validate, s *services.RentalService) *RentalHandler {
    return &RentalHandler{
        validate: v,
		service: s,
    }
}

// (TODO) write better error handling
func (h *RentalHandler) Post(c *fiber.Ctx) error {
	var rental models.Rental

	if err := c.BodyParser(&rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	if err := h.validate.Struct(rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	if err := h.service.Post(rental); err != nil {
		statusCode := fiber.StatusBadRequest
		if errors.Is(err, services.ErrUserAlreadyHosting) {
			statusCode = fiber.StatusConflict
		}

		return c.Status(statusCode).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"message": "Rental posted successfully",
	})
}

func (h *RentalHandler) Get(c *fiber.Ctx) error {
	address := c.Query("address")
	if address == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "address if required",
		})
	}

	re, err := h.service.Get(address)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusOK).JSON(re)
}

func (h *RentalHandler) GetAll(c *fiber.Ctx) error {
	rentals, err := h.service.GetAll()
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"rentals": rentals,
	})
}

