package repos

import "errors"

var ErrDuplicateEmail = errors.New("email already exists")
var ErrUserNotFound = errors.New("user not found")
var ErrListingNotFound = errors.New("listing not found")