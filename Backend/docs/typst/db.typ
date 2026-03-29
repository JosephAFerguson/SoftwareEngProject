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
`sudo mysql` to enter the mysql cli, then run `CREATE DATABASE SoftwareEngProject` \
exit the mysql cli and run `sudo mysql < admin.sql` \
finally, run `sudo mysql < sublease.sql` \

== [ Tests ] <tests>
All of the _.sh_ files are test scripts you can run to make sure certain endpoints work. I have them in _/db_ for now because I was using them to test DB integration. Will probably move these later, but thats what they are. For scripts that require an argument, that argument is the amount of entries to add. The script will automatically populate that many entries to the database.
