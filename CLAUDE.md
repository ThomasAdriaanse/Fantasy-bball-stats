# Fantasy Basketball Scraper Website Commands & Guidelines

## Build and Run Commands
- Run local development server: `flask run` or `python app.py`
- Run with production server: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Docker build: `docker build -t fantasy-scraper-website .`
- Docker run: `docker run -p 5000:5000 fantasy-scraper-website`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports in separate blocks
- **Naming**: Use snake_case for variables and functions, CamelCase for classes
- **Error Handling**: Use specific exception types and handling for ESPN API errors
- **Types**: Use descriptive variable names instead of explicit type annotations
- **Formatting**: Use 4 spaces for indentation, line limit ~100 characters
- **Documentation**: Add docstrings for functions that explain parameters and return values
- **Data Processing**: Use pandas DataFrames for data manipulation and transformation
- **HTML Templates**: Follow Bootstrap/CSS styling in templates directory
- **API Handling**: Handle ESPN API errors explicitly with proper user feedback

## Project Structure
- Flask routes in app.py
- Data processing logic in compare_page/ directory
- Database utilities in db_utils.py
- Frontend templates in templates/ directory
- Static assets in static/ directory