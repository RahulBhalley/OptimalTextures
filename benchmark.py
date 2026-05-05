import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


TOOK_RE = re.compile(r"^Took:\s+([0-9.]+)", re.MULTILINE)


def run_case(args: argparse.Namespace, device: str, size: int, variant: str) -> list[float]:
    env = os.environ.copy()
    if device == "mps" and args.mps_fast_math:
        env["PYTORCH_MPS_FAST_MATH"] = "1"
        env["PYTORCH_MPS_PREFER_METAL"] = "1"

    command = [
        sys.executable,
        "optex.py",
        "--device",
        device,
        "--style",
        args.style,
        "--size",
        str(size),
        "--passes",
        str(args.passes),
        "--iters",
        str(args.iters),
        "--output_dir",
        str(args.output_dir),
    ]
    memory_format = "auto" if variant == "auto" else "channels_last" if variant == "channels_last" else "contiguous"
    command.extend(["--memory_format", memory_format])
    if variant == "compile":
        command.append("--compile")
    elif variant == "script":
        command.append("--script")
    if args.preset:
        command.extend(["--preset", args.preset])
    if args.content:
        command.extend(["--content", args.content, "--content_strength", str(args.content_strength)])
    if args.no_multires:
        command.append("--no_multires")
    if args.no_pca:
        command.append("--no_pca")
    if args.hist_mode:
        command.extend(["--hist_mode", args.hist_mode])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])

    timings = []
    for repeat in range(args.repeats):
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=args.timeout)
        match = TOOK_RE.search(result.stdout)
        if match is None:
            raise RuntimeError(f"Could not parse timing from optex output:\n{result.stdout}\n{result.stderr}")
        timing = float(match.group(1))
        timings.append(timing)
        print(f"{device:>3} {variant:<13} size={size:<4} repeat={repeat + 1:<2} took={timing:.4f}s", flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OptimalTextures inference on CPU and Apple MPS.")
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"], choices=["cpu", "mps"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "auto", "channels_last", "compile", "script"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--style", default="style/lava-small.jpg")
    parser.add_argument("--content", default="content/rocket.jpg")
    parser.add_argument("--content_strength", type=float, default=0.2)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--preset", choices=["fast", "balanced", "quality"], default=None)
    parser.add_argument("--hist_mode", default="chol", choices=["sym", "pca", "chol", "cdf"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_multires", action="store_true", default=True)
    parser.add_argument("--use_multires", action="store_false", dest="no_multires")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--mps_fast_math", action="store_true", default=True)
    parser.add_argument("--no_mps_fast_math", action="store_false", dest="mps_fast_math")
    parser.add_argument("--output_dir", type=Path, default=Path("benchmark_output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark configuration")
    print(f"devices={args.devices} variants={args.variants} sizes={args.sizes} repeats={args.repeats}")
    print(
        f"preset={args.preset} passes={args.passes} iters={args.iters} "
        f"no_multires={args.no_multires} no_pca={args.no_pca}"
    )
    print(f"mps_fast_math={args.mps_fast_math} output_dir={args.output_dir}")
    print()

    all_results: dict[tuple[str, str, int], list[float]] = {}
    for size in args.sizes:
        for device in args.devices:
            for variant in args.variants:
                try:
                    all_results[(device, variant, size)] = run_case(args, device, size, variant)
                except subprocess.TimeoutExpired:
                    print(f"{device:>3} {variant:<13} size={size:<4} timed out after {args.timeout}s")
                except subprocess.CalledProcessError as exc:
                    print(f"{device:>3} {variant:<13} size={size:<4} failed with exit code {exc.returncode}")
                    print(exc.stderr.strip())
                print()

    print("Summary")
    for (device, variant, size), timings in all_results.items():
        mean = statistics.mean(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{device:>3} {variant:<13} size={size:<4} mean={mean:.4f}s stdev={stdev:.4f}s runs={timings}")


if __name__ == "__main__":
    main()
