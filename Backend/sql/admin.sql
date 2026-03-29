CREATE USER 'sublease-admin'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON SoftwareEngProject.* TO 'sublease-admin'@'localhost';
FLUSH PRIVILEGES;

