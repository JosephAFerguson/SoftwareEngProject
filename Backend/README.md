# Setup for Local Backend Environment

The backend can be split up into three distinct pieces, all of which have a short explanation below:

---

1. The API Middleware/Backend. This is written in Go using the Fiber framework. Fiber is very similar to 
ExpressJS if you have used that and follows REST architecture. After cloning the repo, you can run
`go run server.go` in the Backend directory to run this service. All the modules are automatically installed from 
the go.mod file when running the *go run* command.

This is using architecture similar to MVC (Model, View, Controller). The main server is ran with the *server.go* file
listed in the backend root directory. Inside the *internals* folder are four sub-directories: `controllers, middleware,
models, and routes`.

> 1. Controllers are for the route handlers, or in other words the business logic that runs for each route.
>
> 2. Middleware is similar to controllers, except it is logic ran for groups instead of routes. While middleware 
> code is different than controllers as it runs before a given route. For example:
>
> `func UserRoutes(router fiber.Router) {`<br>
> &nbsp;&nbsp;&nbsp;&nbsp;`userGroup := router.Group("/:user", middleware.UserExistsMiddleware)` <br>
> &nbsp;&nbsp;&nbsp;&nbsp;`userGroup.Get("/settings", controllers.GetUserInfo) `<br>
> `}`
>
> In this example, any route that has */:user* (wildcard for user value) in it will run the middleware 
> "UserExistsMiddleware" function. However, only the route within the userGroup (meaning it has */:user* in the 
> route) with the */settings* endpoint will call the controller "GetUserInfo" function.
>
> 3. Models are used as structs or types to format data entry into the database.
> 
> 4. Routes are groups of routes separated by common function for a more simple structure. For instance, the above
> example is in a *users.go* file in the routes directory.

To access api endpoints from the frontend, simply fetch `localhost:3000/api/v1/{route}` with the backend server
running. I will add further documentation to the different routes and route groups once they are more developed 
(In *docs* file. Again, no DB integration yet so routes don't do anything).

---

2. The DB. This is to be setup by Jason. TBD

---

3. Ameya is working on the ML model. Feel free to change what libraries or virtual environment we use. 

The Python ML model. Currently, we are using Anaconda as the virtual environment. After installing Anaconda, go
to the *model* directory and run the following command:

`conda create -p ./.conda python=3.13 &&
conda activate ./.conda &&
conda install -c conda-forge scikit-learn &&
pip install -r requirements.txt`

To test that this all successful, you can run this command:

`python -c "import flask, torch, sklearn; print(f'Flask: {flask.__version__}\nPyTorch: {torch.__version__}\nSklearn: {sklearn.__version__}')"`


---

If you are working on any particular part of the backend and want to add easy documentation to the README, feel 
free to change or append anything to this document.

