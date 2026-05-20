"""Uvicorn runner for FLUX.1-dev Studio."""

import os

import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()
    port = int(os.environ.get("FRONTEND_PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
