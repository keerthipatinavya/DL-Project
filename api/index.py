import os
import sys

# Ensure root project directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app

# Export FastAPI app for Vercel Serverless Function handler
app = app
