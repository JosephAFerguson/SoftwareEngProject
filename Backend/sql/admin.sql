CREATE USER 'sublease-admin'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON softwareengproject.* TO 'sublease-admin'@'localhost';
FLUSH PRIVILEGES;

