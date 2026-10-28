"""Train the Random Forest model on a compatible energy database.

The input workbook may have any feature columns, provided it includes the
numeric target column. Supply another workbook with ``--input``. Training uses
a random 80:20 row split, following the referenced GitHub example.

Run from the project root with::

    .venv/bin/python .venv/ML/RandomForest/train_IS2cleaned_data.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "ISDBcleaned_DB.xlsx").is_file()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_INPUT = PROJECT_ROOT / "ISDBcleaned_DB.xlsx"
TRAINING_SCRIPT = SCRIPT_PATH.with_name("ML")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Excel workbook containing the target column")
    parser.add_argument("--target", default="primary_energy_consumption", help="Target column to predict")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing")
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Path for the saved model bundle (default: artifacts/random_forest_<input-name>_model.joblib)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Excel file not found: {args.input}")
    headings = pd.read_excel(args.input, nrows=0).columns.tolist()
    if args.target not in headings:
        raise ValueError(
            f"Input workbook is missing target column {args.target!r}. "
            "Other feature columns may vary."
        )

    model_output = args.model_output or (
        PROJECT_ROOT / "artifacts" / f"random_forest_{args.input.stem}_model.joblib"
    )

    # Reuse the main Random Forest training implementation while supplying
    # this workbook and a separate default artifact path.
    sys.argv = [
        str(TRAINING_SCRIPT),
        "--input", str(args.input),
        "--target", args.target,
        "--test-size", str(args.test_size),
        "--model-output", str(model_output),
    ]
    runpy.run_path(str(TRAINING_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
