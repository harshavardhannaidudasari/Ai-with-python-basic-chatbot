"""Entry point: run the interactive terminal chatbot.

    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chatbot.cli import run  # noqa: E402

if __name__ == "__main__":
    run()
