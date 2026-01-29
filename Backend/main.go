package main

import (
	"log"
	"database/sql"

	"github.com/gofiber/fiber/v2"
	//"github.com/go-sql-driver/mysql" use for config
	"github.com/go-playground/validator/v10"

	"github.com/JosephAFerguson/SoftwareEngProject/internals/routes"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/handlers"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/services"
	"github.com/JosephAFerguson/SoftwareEngProject/internals/repos"
)

const PORT = "3000"

// could use refactoring asp. maybe move all non-fiber components to an app.go?
func main() {
	app := fiber.New()

	api := app.Group("/api")

	v1 := api.Group("/v1", func(c *fiber.Ctx) error {
		c.Set("Version", "v1")
		return c.Next()
	})

	// DB setup template code. Change user, password, dbname to actual
	// MYSQL driver can be used to setup MYSQL-specific config
	db, err := sql.Open("mysql", "user:password@/dbname")
	if err != nil {
		panic(err)
	}

	//Struct Validator
	validate := validator.New(validator.WithRequiredStructEnabled())

	//Admin Group
	adminHandler := handlers.NewAdminHandler(validate)
	routes.AdminRoutes(v1, adminHandler)

	//Auth Group
	authRepo := repos.NewAuthRepo(db)
	authService := services.NewAuthService(authRepo)
	authHandler := handlers.NewAuthHandler(validate, authService)
	routes.AuthRoutes(v1, authHandler)

	//User Group
	//userHandler := handlers.NewUserHandler(validate)
	//routes.UserRoutes(v1, userHandler)

	log.Fatal(app.Listen(":" + PORT))
}

