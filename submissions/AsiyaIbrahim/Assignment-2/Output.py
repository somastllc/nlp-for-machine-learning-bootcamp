"""
Run preprocessing_practice.py and save all printed results to output.txt.
"""

import os
from pathlib import Path
import subprocess
import sys


FOLDER = Path(__file__).resolve().parent
PRACTICE_SCRIPT = FOLDER / "preprocessing_practice.py"
OUTPUT_FILE = FOLDER / "output.txt"


def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(PRACTICE_SCRIPT)],
        cwd=FOLDER,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    output = result.stdout
    if result.stderr:
        output += "\n\n[STDERR]\n" + result.stderr

    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"Output saved to: {OUTPUT_FILE}")

    if result.returncode != 0:
        print(f"The practice script failed with exit code {result.returncode}.")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
