# Sample Project

This is a simple sample project to demonstrate code analysis capabilities of the Talk to Code application.

## Overview

The project implements a basic data management system with users and products. It provides functionality for:

- Creating and managing users
- Creating and managing products
- Loading and saving data to JSON files
- Configuration management

## Project Structure

- `main.py`: Entry point of the application
- `models/`: Contains data models
  - `user.py`: User model
  - `product.py`: Product model
- `utils/`: Utility functions
  - `helper.py`: Helper utilities for the application
- `config.json`: Application configuration
- `data/`: Directory for data storage (created at runtime)

## Usage

Run the application with:

```bash
python main.py
```

This will:
1. Load configuration
2. Initialize the data manager
3. Create sample data if none exists
4. Display the users and products

## Development

The project demonstrates:
- Object-oriented design with class inheritance
- File I/O operations
- Error handling
- Configuration management
- Data processing

## License

This is a sample project for demonstration purposes only. 