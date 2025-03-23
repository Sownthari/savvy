import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = "https://sandbox.plaid.com"  # Change to "https://development.plaid.com" for live data
