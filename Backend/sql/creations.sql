-- Test Users
-- Password is hashed (example: using a simple hash for testing - in production use bcrypt)
INSERT INTO users (email, password, name, gender, profile_photo) VALUES
('fergujp@mail.uc.edu', 'debug', 'Joseph Ferguson', 'M', 'users/fergujp.png'),
('bellerjn@mail.uc.edu', 'debug', 'Jason Bellerjeau', 'F', 'users/bellerjn.png'),
('gerverge@mail.uc.edu', 'debug', 'Garrett Gerver', 'M', 'users/gerverge.png'),
('khairyas@mail.uc.edu', 'debug', 'Anas Khairy', 'F', 'users/khairyas.png');

-- Test Listings
-- Photos are stored as JSON array of S3 object keys
INSERT INTO listings (user_id, address, title, price, sqft, roommates, bednum, bathnum, pet_friendly, available_from, available_to, photos) VALUES
(1, '123 Main Street, Cincinnati, OH 45202', 'Uptown Place', 1200, 1200, 2, 3, 2, true, '2026-04-01', '2027-03-31', '["unit-1/photo1.png", "unit-1/photo2.png", "unit-1/photo3.png"]'),
(2, '456 Oak Avenue, Cincinnati, OH 45205', 'Cozy Place', 950, 900, 1, 2, 1, false, '2026-04-01', '2027-03-31', '["unit-2/photo1.png"]'),
(3, '789 Elm Drive, Cincinnati, OH 45219', 'Big Place', 1400, 1500, 3, 4, 2.5, true, '2026-05-01', '2027-04-30', '["unit-3/photo1.png", "unit-3/photo2.png"]'),
(4, '321 Maple Lane, Cincinnati, OH 45202', 'Upset Place', 800, 800, 0, 1, 1, false, '2026-04-15', '2027-04-14', '["unit-4/photo1.png"]');

-- Test Reviews
INSERT INTO reviews (listing_id, user_id, rating, review_text)
SELECT l.listing_id, u.user_id, 5, 'Great location and friendly roommates! Very clean and well-maintained apartment.'
FROM listings l
JOIN users u ON u.email = 'fergujp@mail.uc.edu'
WHERE l.address = '123 Main Street, Cincinnati, OH 45202' AND l.title = 'Uptown Place'
UNION ALL
SELECT l.listing_id, u.user_id, 4, 'Good value for the price. Communication with landlord was responsive.'
FROM listings l
JOIN users u ON u.email = 'khairyas@mail.uc.edu'
WHERE l.address = '123 Main Street, Cincinnati, OH 45202' AND l.title = 'Uptown Place'
UNION ALL
SELECT l.listing_id, u.user_id, 5, 'Excellent small unit, perfect for a single person. Quiet neighborhood.'
FROM listings l
JOIN users u ON u.email = 'khairyas@mail.uc.edu'
WHERE l.address = '456 Oak Avenue, Cincinnati, OH 45205' AND l.title = 'Cozy Place'
UNION ALL
SELECT l.listing_id, u.user_id, 3, 'Spacious but had some maintenance issues. Landlord was helpful fixing things.'
FROM listings l
JOIN users u ON u.email = 'fergujp@mail.uc.edu'
WHERE l.address = '789 Elm Drive, Cincinnati, OH 45219' AND l.title = 'Big Place';

-- Test Preferences
INSERT INTO preferences (user_id, preferred_location, budget_min, budget_max, preferred_roommates, preferred_bednum, preferred_bathnum) VALUES
(1, 'Cincinnati, OH', 800, 1500, 2, 2, 1.5),
(4, 'Cincinnati, OH', 1000, 1800, 1, 2, 1),
(3, 'Cincinnati, OH', 900, 1600, 0, 1, 1);
