"""Run the data steps from the project checkout."""

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("step", choices=["prepare", "train", "report", "all"])
step = parser.parse_args().step
steps = ["prepare", "train", "report"] if step == "all" else [step]
for name in steps:
    runpy.run_module(f"hotel_booking.{name}", run_name="__main__")
