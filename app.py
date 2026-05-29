"""
Risk Manager – Emerging Risks Dashboard
========================================
Launch with:
    python app.py

This will:
  1. Install dependencies (if not already installed)
  2. Start the Streamlit dashboard in your browser
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install requirements if streamlit isn't available."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Installing dependencies (first run only)...")
        req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
        )
        print("Dependencies installed successfully.\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    install_dependencies()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "theme_engine/web_dashboard.py",
         "--server.headless", "false"],
    )
