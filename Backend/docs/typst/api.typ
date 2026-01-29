#set page(paper: "a5")
#set heading(bookmarked: false)

#show link: set text(fill: blue, weight: 700)
#show link: underline

= API v1 Documentation

This document serves as documentation for the available routes for the backend API. This document is made using Typst--a programming language for PDF--so if you make changes to the API and want to document them, edit the *_docs.typ_* file.

The API is currently in v1, which denotes using the Fiber framework for RESTful API. The base URL for the v1 API, which is used for all routes, is: *_localhost:3000/api/v1_*. This will be implied for all routes in this document.

Endpoints are separated into groups. Group is the technical term used by Fiber to denote common base URLs (e.g. /user for all user-related requests). Both in the source code and in this document, routes will be separated by group.

== [ /auth ] <auth>
*_/auth_* is used for routes authenticating users into the site.

*[/auth/signup]* \
*_/auth/signup_* is used to add a new user to the users table. \
  req: `{"email": <email>, "password": <password>}` \
  res: `{"res": <code>, "msg": <message>}`

*[/auth/login]* \
*_/auth/login_* is used to validate the user against the users table. \
  req: `{"email": <email>, "password": <password>}` \
  res: `{"res": <code>, "msg": <message>}`

== [ /admin ] <admin>
*_/admin_* is used for internal routes (e.g. health checks, debugging).

*[/admin/health]* \
*_/admin/health_* is used as a "check health" for the API.

=== _Routes are currently WIP, and thus very few routes are shown. This document currently serves as a template for future documentation._

