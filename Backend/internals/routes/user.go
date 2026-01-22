package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/middleware"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/controllers"
)

func UserRoutes(router fiber.Router) {
	userGroup := router.Group("/:user", middleware.UserExistsMiddleware) 

	userGroup.Get("/", controllers.GetUserInfo)
}

