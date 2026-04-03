package handlers

import (
	"errors"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/models"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
)

type AuthHandler struct {
    validate *validator.Validate
	service *services.AuthService
}

func NewAuthHandler(v *validator.Validate, s *services.AuthService) *AuthHandler {
    return &AuthHandler{
        validate: v,
		service: s,
    }
}

func (h *AuthHandler) Signup(c *fiber.Ctx) error {
	var user models.User

	if err := c.BodyParser(&user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if err := h.validate.Struct(user); err != nil {
		validationErrors := parseValidationErrors(err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Validation failed",
			"details": validationErrors,
		})
	}

	userID, err := h.service.Signup(user)
	if err != nil {
		if errors.Is(err, repos.ErrDuplicateEmail) {
			return c.Status(fiber.StatusConflict).JSON(fiber.Map{
				"error": "Email already registered",
				"details": "This email address is already associated with an account",
			})
		}

		// MySQL or other database error
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Account creation failed",
			"details": "An error occurred while creating your account. Please try again later",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"user_id": userID,
	})
}

func (h *AuthHandler) Login(c *fiber.Ctx) error {
	var user models.User

	if err := c.BodyParser(&user); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if err := h.validate.Struct(user); err != nil {
		validationErrors := parseValidationErrors(err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Validation failed",
			"details": validationErrors,
		})
	}

	userID, err := h.service.Login(user)
	if err != nil {
		errMsg := err.Error()
		
		// Check for "no such email" error
		if strings.Contains(errMsg, "no such email") {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid credentials",
				"details": "Email or password is incorrect",
			})
		}
		
		// Check for incorrect password error
		if strings.Contains(errMsg, "Incorrect Password") {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid credentials",
				"details": "Email or password is incorrect",
			})
		}
		
		// Database or service error
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Login failed",
			"details": "An error occurred during login. Please try again later",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"user_id": userID,
	})
}

// parseValidationErrors extracts field validation errors from validator.ValidationErrors
func parseValidationErrors(err error) map[string]string {
	errMap := make(map[string]string)
	
	if validationErrs, ok := err.(validator.ValidationErrors); ok {
		for _, fieldError := range validationErrs {
			field := fieldError.Field()
			tag := fieldError.Tag()
			
			switch tag {
			case "required":
				errMap[field] = field + " is required"
			case "email":
				errMap[field] = field + " must be a valid email address"
			case "max":
				errMap[field] = field + " exceeds maximum length"
			case "gt":
				errMap[field] = field + " must be greater than 0"
			default:
				errMap[field] = field + " failed validation: " + tag
			}
		}
	}
	
	return errMap
}

