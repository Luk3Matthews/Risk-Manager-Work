"""
Risk Manager – Emerging Risks Dashboard
========================================
HOW TO RUN:
    python app.py

That's it. This script handles everything automatically:
  1. Creates a virtual environment (first time only)
  2. Installs all dependencies (first time only)
  3. Launches the dashboard in your browser
"""

import subprocess
import sys
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, ".venv")
REQ_FILE = os.path.join(ROOT, "requirements.txt")

if platform.system() == "Windows":
    VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")


def ensure_venv():
    """Create virtual environment if it doesn't exist."""
    if os.path.isfile(VENV_PYTHON):
        return
    print("=" * 50)
    print("  First-time setup: creating virtual environment...")
    print("=" * 50)
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR, "--upgrade-deps"])
    print("  Virtual environment created.\n")


def ensure_dependencies():
    """Install requirements into the venv if streamlit isn't available."""
    # Make sure pip is available in the venv
    pip_check = subprocess.run(
        [VENV_PYTHON, "-m", "pip", "--version"], capture_output=True
    )
    if pip_check.returncode != 0:
        print("  Bootstrapping pip in virtual environment...")
        subprocess.check_call([VENV_PYTHON, "-m", "ensurepip", "--upgrade"])

    result = subprocess.run(
        [VENV_PYTHON, "-c", "import streamlit"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("  Installing dependencies (this may take a minute)...")
        subprocess.check_call(
            [VENV_PYTHON, "-m", "pip", "install", "-r", REQ_FILE,
             "--quiet", "--disable-pip-version-check"],
        )
        print("  Dependencies installed successfully.\n")


def launch_dashboard():
    """Start the Streamlit dashboard."""
    print("=" * 50)
    print("  Launching Emerging Risks Dashboard")
    print("  URL: http://localhost:8501")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()
    subprocess.run(
        [VENV_PYTHON, "-m", "streamlit", "run",
         os.path.join("theme_engine", "web_dashboard.py"),
         "--server.headless", "false"],
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        ensure_venv()
        ensure_dependencies()
        launch_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during setup: {e}")
        print("Make sure Python 3.12+ is installed: https://www.python.org/downloads/")
        input("Press Enter to exit...")
