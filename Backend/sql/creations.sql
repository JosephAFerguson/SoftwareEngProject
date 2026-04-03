-- Test Users (10 total)
-- Password is hashed SHA-256 of 'admin'
INSERT INTO users (email, password, name, gender, profile_photo) VALUES
('fergujp@mail.uc.edu',  '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Joseph Ferguson',  'M', 'users/fergujp.png'),
('bellerjn@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Jason Bellerjeau', 'M', 'users/bellerjn.png'),
('gerverge@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Garrett Gerver',   'M', 'users/gerverge.png'),
('khairyas@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Anas Khairy',      'M', 'users/khairyas.png'),
('smithaj@mail.uc.edu',  '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Amanda Smith',     'F', NULL),
('johnstmk@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Mike Johnston',    'M', NULL),
('patelrv@mail.uc.edu',  '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Ravi Patel',       'M', NULL),
('nguyenlt@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Linh Nguyen',      'F', NULL),
('harrisbd@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Brittany Harris',  'F', NULL),
('obrienct@mail.uc.edu', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'Connor OBrien',    'M', NULL);

-- Test Listings (one per user, 2-3 photos each)
INSERT INTO listings (user_id, address, title, price, sqft, roommates, bednum, bathnum, pet_friendly, available_from, available_to, photos) VALUES
(1,  '123 Main Street, Cincinnati, OH 45202',          'Uptown Place',         1200, 1200, 2, 3, 2.0, true,  '2026-04-01', '2027-03-31', '["unit-1/photo1.png", "unit-1/photo2.png", "unit-1/photo3.png"]'),
(2,  '456 Oak Avenue, Cincinnati, OH 45205',           'Cozy Place',            950,  900, 1, 2, 1.0, false, '2026-04-01', '2027-03-31', '["unit-2/photo1.png", "unit-1/photo1.png"]'),
(3,  '789 Elm Drive, Cincinnati, OH 45219',            'Big Place',            1400, 1500, 3, 4, 2.5, true,  '2026-05-01', '2027-04-30', '["unit-3/photo1.png", "unit-3/photo2.png"]'),
(4,  '321 Maple Lane, Cincinnati, OH 45202',           'Quiet Studio',          800,  800, 0, 1, 1.0, false, '2026-04-15', '2027-04-14', '["unit-4/photo1.png", "unit-1/photo2.png"]'),
(5,  '550 Vine Street, Cincinnati, OH 45202',          'Vine Street Loft',     1350, 1100, 1, 2, 1.5, true,  '2026-05-01', '2027-04-30', '["unit-5/photo1.png", "unit-5/photo2.png"]'),
(6,  '88 Reading Road, Cincinnati, OH 45215',          'Reading Road Retreat',  875,  750, 1, 1, 1.0, false, '2026-04-20', '2027-04-19', '["unit-6/photo1.png"]'),
(7,  '1200 Gilbert Avenue, Cincinnati, OH 45202',      'Gilbert Garden Unit',  1100, 1050, 2, 2, 1.0, true,  '2026-06-01', '2027-05-31', '["unit-7/photo1.png"]'),
(8,  '33 West McMillan Street, Cincinnati, OH 45219',  'McMillan Modern',      1250, 1000, 1, 2, 2.0, false, '2026-04-01', '2027-03-31', '["unit-8/photo1.png"]'),
(9,  '740 Ludlow Avenue, Cincinnati, OH 45220',        'Clifton Charm',         990,  870, 2, 2, 1.0, true,  '2026-05-15', '2027-05-14', '["unit-9/photo1.png"]'),
(10, '2200 Victory Parkway, Cincinnati, OH 45206',     'Victory View',         1175,  960, 1, 2, 1.5, false, '2026-04-01', '2027-03-31', '["unit-10/photo1.png"]');

-- Test Reviews
INSERT INTO reviews (listing_id, user_id, rating, review_text)
SELECT l.listing_id, u.user_id, 5, 'Great location and friendly roommates! Very clean and well-maintained apartment.'
FROM listings l JOIN users u ON u.email = 'fergujp@mail.uc.edu'
WHERE l.address = '123 Main Street, Cincinnati, OH 45202' AND l.title = 'Uptown Place'
UNION ALL
SELECT l.listing_id, u.user_id, 4, 'Good value for the price. Communication with landlord was responsive.'
FROM listings l JOIN users u ON u.email = 'khairyas@mail.uc.edu'
WHERE l.address = '123 Main Street, Cincinnati, OH 45202' AND l.title = 'Uptown Place'
UNION ALL
SELECT l.listing_id, u.user_id, 5, 'Excellent small unit, perfect for a single person. Quiet neighborhood.'
FROM listings l JOIN users u ON u.email = 'khairyas@mail.uc.edu'
WHERE l.address = '456 Oak Avenue, Cincinnati, OH 45205' AND l.title = 'Cozy Place'
UNION ALL
SELECT l.listing_id, u.user_id, 3, 'Spacious but had some maintenance issues. Landlord was helpful fixing things.'
FROM listings l JOIN users u ON u.email = 'fergujp@mail.uc.edu'
WHERE l.address = '789 Elm Drive, Cincinnati, OH 45219' AND l.title = 'Big Place'
UNION ALL
SELECT l.listing_id, u.user_id, 4, 'Love the loft vibe! Great natural light and close to everything on Vine.'
FROM listings l JOIN users u ON u.email = 'bellerjn@mail.uc.edu'
WHERE l.address = '550 Vine Street, Cincinnati, OH 45202' AND l.title = 'Vine Street Loft'
UNION ALL
SELECT l.listing_id, u.user_id, 5, 'Super affordable, landlord was great. Would recommend to anyone on a tight budget.'
FROM listings l JOIN users u ON u.email = 'gerverge@mail.uc.edu'
WHERE l.address = '88 Reading Road, Cincinnati, OH 45215' AND l.title = 'Reading Road Retreat'
UNION ALL
SELECT l.listing_id, u.user_id, 4, 'Nice garden-level unit, good for quiet study. A bit dark in the back room.'
FROM listings l JOIN users u ON u.email = 'smithaj@mail.uc.edu'
WHERE l.address = '1200 Gilbert Avenue, Cincinnati, OH 45202' AND l.title = 'Gilbert Garden Unit'
UNION ALL
SELECT l.listing_id, u.user_id, 5, 'Modern finishes, great bathroom, close to UC campus. Highly recommend.'
FROM listings l JOIN users u ON u.email = 'patelrv@mail.uc.edu'
WHERE l.address = '33 West McMillan Street, Cincinnati, OH 45219' AND l.title = 'McMillan Modern'
UNION ALL
SELECT l.listing_id, u.user_id, 3, 'Charming area but parking is tough. Apartment itself is clean and cozy.'
FROM listings l JOIN users u ON u.email = 'johnstmk@mail.uc.edu'
WHERE l.address = '740 Ludlow Avenue, Cincinnati, OH 45220' AND l.title = 'Clifton Charm'
UNION ALL
SELECT l.listing_id, u.user_id, 4, 'Great views of the park and easy highway access. A little noisy at night.'
FROM listings l JOIN users u ON u.email = 'nguyenlt@mail.uc.edu'
WHERE l.address = '2200 Victory Parkway, Cincinnati, OH 45206' AND l.title = 'Victory View';

-- Test Preferences (all 10 users)
INSERT INTO preferences (user_id, preferred_location, budget_min, budget_max, preferred_roommates, preferred_bednum, preferred_bathnum) VALUES
(1,  'Cincinnati, OH', 800,  1500, 2, 2, 1.5),
(2,  'Cincinnati, OH', 700,  1200, 1, 2, 1.0),
(3,  'Cincinnati, OH', 900,  1600, 0, 1, 1.0),
(4,  'Cincinnati, OH', 1000, 1800, 1, 2, 1.0),
(5,  'Cincinnati, OH', 1100, 1600, 1, 2, 1.5),
(6,  'Cincinnati, OH', 700,  1000, 1, 1, 1.0),
(7,  'Cincinnati, OH', 900,  1400, 2, 2, 1.0),
(8,  'Cincinnati, OH', 1000, 1500, 1, 2, 2.0),
(9,  'Cincinnati, OH', 800,  1200, 2, 2, 1.0),
(10, 'Cincinnati, OH', 950,  1400, 1, 2, 1.5);
