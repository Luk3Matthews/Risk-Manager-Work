"""
Main entry point for the VFMC News & Macro Risk Monitor.

Usage:
    # Start the ingestion scheduler only (headless):
    python run.py --ingest

    # Start the Streamlit UI only:
    python run.py --ui

    # Start both (default):
    python run.py
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("news_monitor")


def start_ingestion():
    """Start the background news ingestion scheduler."""
    from src.scheduler import NewsScheduler, load_config

    config = load_config()
    scheduler = NewsScheduler(config)
    scheduler.start()
    return scheduler


def start_ui():
    """Launch the Streamlit UI."""
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501"]
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(description="VFMC News & Macro Risk Monitor")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion scheduler only")
    parser.add_argument("--ui", action="store_true", help="Run Streamlit UI only")
    args = parser.parse_args()

    # Default: run both
    run_ingest = args.ingest or (not args.ingest and not args.ui)
    run_ui = args.ui or (not args.ingest and not args.ui)

    scheduler = None
    ui_proc = None

    try:
        if run_ingest:
            logger.info("Starting news ingestion scheduler...")
            scheduler = start_ingestion()

        if run_ui:
            logger.info("Starting Streamlit UI on port 8501...")
            ui_proc = start_ui()

        # Keep main thread alive
        if ui_proc:
            ui_proc.wait()
        elif scheduler:
            # If no UI, just run until interrupted
            import time
            while True:
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if scheduler:
            scheduler.stop()
        if ui_proc and ui_proc.poll() is None:
            ui_proc.terminate()


if __name__ == "__main__":
    main()
