"""
Risk Manager – Emerging Risks Dashboard
========================================
Launch with:
    python app.py

Or equivalently:
    python -m streamlit run theme_engine/web_dashboard.py
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "theme_engine/web_dashboard.py",
         "--server.headless", "false"],
    )
