"""Shared Google OAuth2 authentication for Calendar and Sheets APIs."""

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Scopes for Calendar (read-only) and Sheets (read/write)
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

# Paths for credentials
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_CREDENTIALS_FILE = _CONFIG_DIR / "google_credentials.json"
_TOKEN_FILE = _CONFIG_DIR / "google_token.json"


def get_credentials() -> Credentials | None:
    """Get valid Google credentials, refreshing or prompting auth as needed."""
    # Service account (production / Cloud Run)
    sa_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if sa_env:
        sa_path = Path(sa_env)
        if sa_path.exists():
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                str(sa_path), scopes=SCOPES
            )

    creds = None

    # Load existing token
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not _CREDENTIALS_FILE.exists():
                print(f"Credentials file not found: {_CREDENTIALS_FILE}")
                print("Please download OAuth credentials from Google Cloud Console")
                print("and save as 'google_credentials.json' in the config folder.")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future use
        with open(_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return creds


def is_authenticated() -> bool:
    """Check if we have valid Google authentication."""
    if not _TOKEN_FILE.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)
        return creds and creds.valid
    except Exception:
        return False


def setup_auth():
    """Interactive setup for Google authentication."""
    print("Setting up Google API authentication...")
    print()

    if not _CREDENTIALS_FILE.exists():
        print("Step 1: Download OAuth credentials")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create an OAuth 2.0 Client ID (Desktop app)")
        print("  3. Download the JSON file")
        print(f"  4. Save it as: {_CREDENTIALS_FILE}")
        print("     (config/google_credentials.json)")
        print()
        input("Press Enter when ready...")

    if not _CREDENTIALS_FILE.exists():
        print("Credentials file still not found. Please try again.")
        return False

    print("\nStep 2: Authorize access")
    print("A browser window will open for you to authorize access...")
    print()

    creds = get_credentials()
    if creds:
        print("\nGoogle authentication successful!")
        return True
    else:
        print("\nGoogle authentication failed.")
        return False


if __name__ == "__main__":
    setup_auth()
