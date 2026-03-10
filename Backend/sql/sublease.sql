CREATE TABLE `buying` (
  `Email` binary(32) DEFAULT NULL,
  `Pricemax` int DEFAULT NULL,
  `Genderpref` varchar(45) DEFAULT NULL,
  `Minsqft` int DEFAULT NULL,
  `Bednumpref` int DEFAULT NULL,
  `Petfriendly` tinyint DEFAULT NULL,
  `Bathnumpref` int DEFAULT NULL,
  `Distancefromcapmus` int DEFAULT NULL,
  KEY `Email_idx` (`Email`)
);
CREATE TABLE `rental` (
  `Address` varchar(75) NOT NULL,
  `Price` int DEFAULT NULL,
  `Sqft` int DEFAULT NULL,
  `Roomates` int DEFAULT NULL,
  `Bednum` int DEFAULT NULL,
  `Bathnum` int DEFAULT NULL,
  PRIMARY KEY (`Address`)
);
CREATE TABLE `reviews` (
  `Reviewsid` int NOT NULL,
  `Reviewtext` varchar(250) DEFAULT NULL,
  `Starnumber` int DEFAULT NULL,
  PRIMARY KEY (`Reviewsid`)
);
CREATE TABLE `users` (
  `Email` binary(32) NOT NULL,
  `Password` binary(32) NOT NULL,
  `Name` varchar(45) DEFAULT NULL,
  `Gender` varchar(45) DEFAULT NULL,
  `Reviewsid` int DEFAULT NULL,
  PRIMARY KEY (`Email`),
  KEY `Reviewsid_idx` (`Reviewsid`),
  CONSTRAINT `Reviewsid` FOREIGN KEY (`Reviewsid`) REFERENCES `reviews` (`Reviewsid`)
);
CREATE TABLE `selling` (
  `Address` varchar(75) DEFAULT NULL,
  `Email` binary(32) DEFAULT NULL,
  KEY `Email_idx` (`Email`),
  KEY `Address_idx` (`Address`),
  CONSTRAINT `Address` FOREIGN KEY (`Address`) REFERENCES `rental` (`Address`),
  CONSTRAINT `Email` FOREIGN KEY (`Email`) REFERENCES `users` (`Email`)
);

