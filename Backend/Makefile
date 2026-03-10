.PHONY: test build dev prod clean

test:
	@echo "Running tests..."
	#go test

build:
	@echo "Building Go executable..."
	mkdir -p build/
	go build -o build/ main.go

dev:
	@echo "Starting dev server..."
	# go run . --env ./env/dev.env

prod:
	@echo "Starting prod server..."
	# go run . --env ./env/prod.env
	
clean:
	rm -rf build/

