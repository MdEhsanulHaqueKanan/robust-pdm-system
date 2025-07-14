from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import the routes to register them with the app
# This import is at the bottom to avoid circular dependencies
from app import routes