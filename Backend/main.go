package main

import (
	"log"
	"database/sql"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	_ "github.com/go-sql-driver/mysql"
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

	app.Use(cors.New(cors.Config{
		AllowOrigins: "http://localhost:5173",
		AllowMethods: "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders: "Origin, Content-Type, Accept, Authorization",
	}))

	api := app.Group("/api")

	v1 := api.Group("/v1", func(c *fiber.Ctx) error {
		c.Set("Version", "v1")
		return c.Next()
	})

	// DB setup template code. Change user, password, dbname to actual
	// MYSQL driver can be used to setup MYSQL-specific config
	db, err := sql.Open("mysql", "sublease-admin:password@/SoftwareEngProject")
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

	//Rental Group
	rentalRepo := repos.NewRentalRepo(db)
	rentalService := services.NewRentalService(rentalRepo)
	rentalHandler := handlers.NewRentalHandler(validate, rentalService)
	routes.RentalRoutes(v1, rentalHandler)

	//Profile Group
	profileRepo := repos.NewProfileRepo(db)
	profileService := services.NewProfileService(profileRepo)
	profileHandler := handlers.NewProfileHandler(validate, profileService)
	routes.ProfileRoutes(v1, profileHandler)

	//Preference Group
	preferenceRepo := repos.NewPreferenceRepo(db)
	preferenceService := services.NewPreferenceService(preferenceRepo)
	preferenceHandler := handlers.NewPreferenceHandler(validate, preferenceService)
	routes.PreferenceRoutes(v1, preferenceHandler)

	//Recommendation Group
	recommendationRepo := repos.NewRecommendationRepo(db)
	recommendationService := services.NewRecommendationService(preferenceRepo, rentalRepo, recommendationRepo)
	recommendationHandler := handlers.NewRecommendationHandler(recommendationService)
	routes.RecommendationRoutes(v1, recommendationHandler)

	log.Fatal(app.Listen(":" + PORT))
}

