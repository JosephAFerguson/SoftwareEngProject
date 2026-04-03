package handlers

import (
	"errors"

	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
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

func (h *RentalHandler) Post(c *fiber.Ctx) error {
	var rental models.Rental

	if err := c.BodyParser(&rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if err := h.validate.Struct(rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Validation failed",
			"details": parseValidationErrors(err),
		})
	}

	if err := h.service.Post(rental); err != nil {
		if errors.Is(err, services.ErrUserAlreadyHosting) {
			return c.Status(fiber.StatusConflict).JSON(fiber.Map{
				"error": "User already has listing",
				"details": "This user already has an active listing",
			})
		}

		if errors.Is(err, repos.ErrUserNotFound) {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"error": "User not found",
				"details": "Cannot create a listing for a user that does not exist",
			})
		}

		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Listing creation failed",
			"details": "An error occurred while creating the listing. Please try again later",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"message": "Rental posted successfully",
	})
}

func (h *RentalHandler) Update(c *fiber.Ctx) error {
	var rental models.Rental

	if err := c.BodyParser(&rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if rental.ListingID == 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Validation failed",
			"details": []fiber.Map{{
				"field": "listing_id",
				"error": "listing_id is required",
			}},
		})
	}

	if err := h.validate.Struct(rental); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Validation failed",
			"details": parseValidationErrors(err),
		})
	}

	if err := h.service.Update(rental); err != nil {
		if errors.Is(err, repos.ErrListingNotFound) {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"error": "Listing not found",
				"details": "Cannot update a listing that does not exist for this user",
			})
		}

		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Listing update failed",
			"details": "An error occurred while updating the listing. Please try again later",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"message": "Listing updated successfully",
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

