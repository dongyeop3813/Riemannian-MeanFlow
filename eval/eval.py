import argparse
import subprocess
import sys

from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def call_sample_script(args, sample_dir):
    cmd = [
        "python",
        "src/eval/gen_sample_from_ckpt.py",
        "--ckpt_dir",
        str(args.ckpt_dir),
        "--sample_dir",
        str(sample_dir),
        "--split",
        str(args.split),
        "--seed",
        str(args.seed),
        "--epoch",
        str(args.epoch),
    ]

    if args.use_integration:
        cmd.append("--use_integration")

    eval_log = Path(sample_dir) / "eval.log"
    print(f"Log file: {eval_log}")
    print(f"Command: {' '.join(cmd)}")

    with open(eval_log, "w") as f:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
        if result.returncode != 0:
            f.write(f"\nSample script exited with code {result.returncode}\n")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )


def call_eval_script(args, sample_dir):
    cmd = [
        "python",
        "src/eval/eval_script.py",
        "--sample_dir",
        str(sample_dir),
    ]

    if args.project_to_onehot:
        cmd.append("--project_to_onehot")

    eval_log = Path(sample_dir) / "eval.log"
    print(f"Command: {' '.join(cmd)}")
    with open(eval_log, "a") as f:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
        if result.returncode != 0:
            f.write(f"\nEval script exited with code {result.returncode}\n")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument(
        "--skip_sample",
        action="store_true",
        help="Skip sample generation (default: run sample generation)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after generating samples (default: run evaluation)",
    )
    parser.add_argument(
        "--use_integration",
        action="store_true",
        help="Use integration for sampling",
    )
    parser.add_argument(
        "--no-projection",
        dest="project_to_onehot",
        action="store_false",
        help="Do not project to one-hot encoding for evaluation (default: project to one-hot)",
    )
    parser.set_defaults(project_to_onehot=True)

    args = parser.parse_args()

    seed = args.seed

    dirname = f"epoch_{args.epoch}_seed_{seed}"
    dirname += "_integration" if args.use_integration else ""
    dirname += "_project" if args.project_to_onehot else ""

    args.ckpt_dir = Path(args.ckpt_dir).resolve()
    sample_dir = Path(args.ckpt_dir) / "eval" / dirname
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = sample_dir.resolve()

    # Default: run sample & eval unless skip flags are set
    if not args.skip_sample:
        call_sample_script(args, sample_dir)

    if not args.skip_eval:
        call_eval_script(args, sample_dir)

        # Read and print evaluation results from CSV
        eval_csv_path = sample_dir / "eval.csv"
        if eval_csv_path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluation Results from {eval_csv_path}:")
            print(f"{'='*60}")
            with open(eval_csv_path, "r") as f:
                print(f.read())
            print(f"{'='*60}\n")
        else:
            print(f"\nWarning: No eval.csv found at {eval_csv_path}")
            print("Evaluation may have failed or CSV was not generated.\n")
