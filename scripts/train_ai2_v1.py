"""CLI entrypoint for training the AI2 v1 baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai2.train import train_ai2_v1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DimScan AI2 v1 baseline.")
    parser.add_argument("--job-type", default="single", choices=("single", "group"))
    parser.add_argument("--jobs-root")
    parser.add_argument("--out")
    args = parser.parse_args()

    jobs_root = args.jobs_root or f"datasets/{args.job_type}/jobs"
    out_dir = args.out or f"models/packing/{args.job_type}/ai2_v1"
    report = train_ai2_v1(jobs_root, out_dir, job_type=args.job_type)
    dataset = report["dataset"]
    print(f"included={dataset['included']} skipped={dataset['skipped']}")
    print(f"skip_reasons={dataset['skip_reasons']}")
    print(f"label_summary={report['label_summary']}")
    if report["status"] != "trained":
        print(f"status={report['status']} warnings={report['warnings']}", file=sys.stderr)
        return 1
    print(f"model_path={report['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
