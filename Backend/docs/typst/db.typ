#set page(paper: "a5")
#set heading(bookmarked: false)

#show link: set text(fill: blue, weight: 700)
#show link: underline

= MYSQL Documentation

This document serves as documentation for how to setup the MYSQL database. This document is made using Typst--a programming language for PDF--so if you make changes to the DB and want to document them, edit the *_db.typ_* file.

The schema varies slighly from the ERD, namely that instead of using a sixtwo number to sign in, you provide your full email (to make this school agnostic), adding a password field to the users table, and encrypting both the email and password fields (I only use SHA256, which we can decrypt trivially to get raw email to display back. This is moreso used to combat sql injections).

That being said, for documentation on how the DB is actually configured, you can either view the ERD or inspect your local copy of the DB. This document is for initialization.

== [ Initialization ] <init>
*_TLDR_* Copy and paste the following: \
`sudo mysql < admin.sql` \
`sudo mysql` -> `CREATE DATABASE SoftwareEngProject` \
`sudo mysql < sublease.sql` \

*_1._* Run the _admin.sql_ file to create the admin user the backend uses by default (e.g. `sudo mysql < admin.sql`). You can optionally use your own user and change the user field in main.go. I was going to back this an environment variable, but I figured people can just use the provided admin user (hopefully).

*_2._* Either run `sudo mysql` to use your default root/superuser user, or to use the mysql cli with the created admin user do `mysql --user=sublease-admin --password=password`. Once you have signed in and are in the mysql cli, create the database for this project `CREATE DATABASE <dbname>`. To adhere to the main.go file, use `CREATE DATABASE SoftwareEngProject`. However, you can make this whatever _as long as_ you change the appropriate value in main.go.

*_3._* Now that you have created your database, run the command `sudo mysql <dbname> < sublease.sql` where <dbname> is the name of your database. This will insert all of the needed tables from the ERD with the modifications described above.

== [ Tests ] <tests>
All of the _.zsh_ files are test scripts you can run to make sure certain endpoints work. I have them in _/db_ for now because I was using them to test DB integration. Will probably move these later, but thats what they are.

