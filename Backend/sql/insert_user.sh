#!/bin/zsh

curl -X POST http://localhost:3000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"John@gmail.com","password":"john"}'
