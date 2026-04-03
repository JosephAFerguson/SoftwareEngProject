CREATE TABLE `users` (
  `user_id` INT AUTO_INCREMENT PRIMARY KEY,
  `email` VARCHAR(255) NOT NULL UNIQUE,
  `password` CHAR(64) NOT NULL,
  `name` VARCHAR(100),
  `gender` VARCHAR(10),
  `profile_photo` VARCHAR(255)
);

CREATE TABLE `listings` (
  `listing_id` INT AUTO_INCREMENT PRIMARY KEY,
  `user_id` INT NOT NULL,
  `title` VARCHAR(255) NOT NULL,
  `address` VARCHAR(255) NOT NULL,
  `price` INT NOT NULL,
  `sqft` INT,
  `roommates` INT DEFAULT NULL,
  `bednum` INT DEFAULT NULL,
  `bathnum` DECIMAL(3,1) DEFAULT NULL,
  `pet_friendly` BOOLEAN DEFAULT FALSE,
  `available_from` DATE DEFAULT NULL,
  `available_to` DATE DEFAULT NULL,
  `photos` JSON DEFAULT NULL
);

CREATE TABLE `reviews` (
  `review_id` INT AUTO_INCREMENT PRIMARY KEY,
  `listing_id` INT NOT NULL,
  `user_id` INT NOT NULL,
  `rating` TINYINT NOT NULL,
  `review_text` VARCHAR(500)
);

CREATE TABLE `messages` (
  `message_id` INT AUTO_INCREMENT PRIMARY KEY,
  `sender_id` INT NOT NULL,
  `receiver_id` INT NOT NULL,
  `content` TEXT NOT NULL,
  `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE `preferences` (
  `preference_id` INT AUTO_INCREMENT PRIMARY KEY,
  `user_id` INT NOT NULL,
  `preferred_location` VARCHAR(255) NOT NULL,
  `budget_min` INT,
  `budget_max` INT,
  `preferred_roommates` INT,
  `preferred_bednum` INT,
  `preferred_bathnum` DECIMAL(3,1)
);

CREATE TABLE `user_recommendations` (
  `recommendation_id` INT AUTO_INCREMENT PRIMARY KEY,
  `user_id` INT NOT NULL,
  `listing_id` INT NOT NULL,
  `score` FLOAT NOT NULL
);

-- Add foreign key constraints after tables are created
ALTER TABLE `listings` ADD CONSTRAINT `fk_listings_user` FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE;
ALTER TABLE `reviews` ADD CONSTRAINT `fk_reviews_listing` FOREIGN KEY (`listing_id`) REFERENCES `listings`(`listing_id`) ON DELETE CASCADE;
ALTER TABLE `reviews` ADD CONSTRAINT `fk_reviews_user` FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE;
ALTER TABLE `preferences` ADD CONSTRAINT `fk_preferences_user` FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE;
ALTER TABLE `user_recommendations` ADD CONSTRAINT `fk_recommendations_user` FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE;
ALTER TABLE `user_recommendations` ADD CONSTRAINT `fk_recommendations_listing` FOREIGN KEY (`listing_id`) REFERENCES `listings`(`listing_id`) ON DELETE CASCADE;
