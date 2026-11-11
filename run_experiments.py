"""Build and run HFX with the paper's layouts and scheduled private releases."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
SUFFIX = ".exe" if os.name == "nt" else ""
REPLAY = BUILD / ("hfx_replay" + SUFFIX)
CHECK = BUILD / ("hfx_check" + SUFFIX)
METHODS = {"hfx", "gmod_public", "hx_phi"}
FIELDS = ["mode", "dataset", "pattern", "N", "T", "Nmax", "K", "B",
          "memory_kib", "completion", "hot", "seed", "epsilon", "first",
          "trials", "q16", "rows", "gmod_width", "gmod_rows", "input",
          "group", "key", "profile_id"]


def replay_path(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        # Windows paths on another drive have no relative representation.
        return str(Path(path).resolve())


def environment():
    env = {**os.environ, "PYTHONUTF8": "1"}
    mingw = Path(env.get("HFX_MINGW_BIN", "C:/msys64/mingw64/bin"))
    if os.name == "nt" and mingw.is_dir():
        env["PATH"] = str(mingw) + os.pathsep + env.get("PATH", "")
    return env


def build(native=False):
    env = environment()
    name = env.get("CXX", "g++")
    compiler = shutil.which(name, path=env.get("PATH"))
    if compiler is None:
        raise ValueError("A 64-bit GCC/Clang C++20 compiler must be on PATH (or set CXX).")
    BUILD.mkdir(exist_ok=True)
    flags = ["-std=c++20", "-O3"] + (["-march=native"] if native else [])
    for source, target in [("benchmarks/replay.cpp", REPLAY),
                           ("benchmarks/check.cpp", CHECK)]:
        # Relative filenames also work with MinGW's linker in Unicode directories.
        subprocess.run([compiler, *flags, source, "-o", str(target.relative_to(ROOT))],
                       cwd=ROOT, env=env, check=True)
        print(f"Built {target.name}", flush=True)


def load_config(path, job=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if job is not None:
        jobs = value.get("jobs", value)
        if isinstance(jobs, list):
            matches = [entry for entry in jobs if entry["key"] == job]
            if len(matches) != 1:
                raise ValueError(f"Job key must identify one configuration: {job}")
            value = matches[0]
        else:
            value = jobs[job]
    elif "jobs" in value:
        raise ValueError("Select a configuration from paper.json with --job.")
    return value


def validate_config(job):
    layout = job["layout"]
    if job["mode"] not in {"full", "payload", "theory"}:
        raise ValueError("Mode must be full, payload, or theory.")
    for name in ["N", "T", "Nmax", "K", "B", "memory_kib", "trials"]:
        if int(job[name]) != job[name] or int(job[name]) < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if int(job.get("first", 0)) < 0 or int(job["seed"]) < 0:
        raise ValueError("Seed and first trial must be nonnegative.")
    if not math.isfinite(float(job["epsilon"])) or float(job["epsilon"]) <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if float(job.get("delta", 1e-6)) != 1e-6:
        raise ValueError("The supplied layout selector and replay use delta = 1e-6.")
    schedule = "fixed-b" if job["group"] in {"real", "epsilon", "cadence", "history"} else "fixed-k"
    if job.get("schedule", schedule) != schedule:
        raise ValueError("Release schedule does not match the workload group.")
    needs_input = bool(job.get("input_filename")) or job["group"] in {"real", "epsilon", "cadence"}
    prep = 0 if needs_input or job.get("completion", -1) >= 0 else int(job["N"])
    if int(job["T"]) <= prep:
        raise ValueError("Synthetic input must contain updates after initialization.")
    if schedule == "fixed-b":
        actual = (int(job["T"]) - prep + int(job["B"]) - 1) // int(job["B"])
        if actual != int(job["K"]):
            raise ValueError("Release interval does not produce the declared release count K.")
    elif int(job["K"]) > int(job["T"]) - prep:
        raise ValueError("Release count exceeds the available input slots.")
    rows = layout["rows"]
    if not 1 <= len(rows) <= 32 or any(int(m) != m or m < 4 for m in rows):
        raise ValueError("A layout needs 1--32 columns with at least four bits each.")
    q = int(layout["q16"])
    if not 1 <= q <= 16 or (q != 16 and len(rows) == 1):
        raise ValueError("Invalid column routing parameter q16.")
    if not 1 <= int(layout["gmod_width"]) <= 32 or int(layout["gmod_rows"]) < 4:
        raise ValueError("Invalid GMod layout.")
    # A profile must be selected using the same public parameters as its run.
    for name, expected in [("capacity", job["Nmax"]), ("K", job["K"]),
                           ("epsilon", job["epsilon"]),
                           ("mode", "payload" if job["mode"] == "theory" else job["mode"]),
                           ("memory_kib", job["memory_kib"])]:
        if name in layout and layout[name] != expected:
            raise ValueError(f"Layout and workload disagree on {name}.")


def run(job, output, methods="hfx,gmod_public,hx_phi", input_path=None):
    if not REPLAY.is_file():
        raise ValueError("Run 'python run_experiments.py build' first.")
    job = dict(job)
    validate_config(job)
    requested = methods.split(",")
    if not requested or any(name not in METHODS for name in requested):
        raise ValueError("Methods must be drawn from hfx,gmod_public,hx_phi.")
    source = input_path or job.get("input")
    needs_input = bool(job.get("input_filename")) or job["group"] in {"real", "epsilon", "cadence"}
    if needs_input and not source:
        raise ValueError("This configuration needs a processed endpoint stream via --input.")
    if source:
        source = Path(source).resolve()
        with source.open("rb") as stream:
            header = stream.read(16)
        if len(header) != 16 or header[:8] != b"HFXSTRM1":
            raise ValueError("Input must use the HFXSTRM1 endpoint format.")
        if struct.unpack("<Q", header[8:])[0] != int(job["T"]):
            raise ValueError("Input endpoint count differs from the configuration's T.")
        if source.stat().st_size != 16 + 9 * int(job["T"]):
            raise ValueError("Input file length differs from its declared endpoint count.")
    prefix = Path(output).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    layout = job["layout"]
    values = [job["mode"], job["dataset"], job["pattern"], job["N"], job["T"],
              job["Nmax"], job["K"], job["B"], job["memory_kib"],
              job.get("completion", -1), job.get("hot", 0), job["seed"],
              job["epsilon"], job.get("first", 0), job["trials"], layout["q16"],
              ",".join(map(str, layout["rows"])), layout["gmod_width"],
              layout["gmod_rows"], replay_path(source) if source else "-", job["group"],
              job["key"], layout["profile_id"]]
    if any("\t" in str(value) or "\n" in str(value) or "\r" in str(value) for value in values):
        raise ValueError("Configuration fields cannot contain tabs or line breaks.")
    with prefix.with_name(prefix.name + "_job.tsv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, delimiter="\t", quoting=csv.QUOTE_NONE).writerows([FIELDS, values])
    env = environment()
    env["HFX_RUN_METHODS"] = methods
    env.pop("HFX_DIAGNOSTICS", None)
    env["HFX_HX_NOISE_OFFSET"] = "10007"
    subprocess.run([str(REPLAY), replay_path(str(prefix) + "_job.tsv"),
                    replay_path(prefix)],
                   cwd=ROOT, env=env, check=True)
    print(f"Results: {prefix}_summary.csv", flush=True)
    return prefix


def check():
    if not CHECK.is_file():
        raise ValueError("Run 'python run_experiments.py build' first.")
    subprocess.run([str(CHECK)], cwd=ROOT, env=environment(), check=True)
    job = load_config(ROOT / "configs/demo.json")
    prefix = run(job, ROOT / "results/check")
    with Path(str(prefix) + "_summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    with Path(str(prefix) + "_releases.csv").open(newline="", encoding="utf-8") as stream:
        releases = list(csv.DictReader(stream))
    if {row["method"] for row in summaries} != METHODS:
        raise ValueError("The check must evaluate all three methods.")
    for summary in summaries:
        rows = [row for row in releases if row["method"] == summary["method"]]
        if len(rows) != job["K"] or int(summary["release_count"]) != job["K"]:
            raise ValueError("Incorrect release count.")
        if int(summary["processed"]) != job["T"]:
            raise ValueError("Incomplete input replay.")
        if int(summary["state_bytes"]) > job["memory_kib"] * 1024:
            raise ValueError("Complete-state memory budget exceeded.")
        squared = [(float(row["estimate"]) - int(row["truth"])) ** 2 for row in rows]
        rmse = math.sqrt(sum(squared) / len(rows))
        relative = [sq / int(row["truth"]) ** 2 for row, sq in zip(rows, squared) if int(row["truth"]) > 0]
        rmsd = math.sqrt(sum(relative) / len(relative)) if relative else 0
        if not (math.isclose(rmse, float(summary["rmse"]), rel_tol=1e-10, abs_tol=1e-10)
                and math.isclose(rmsd, float(summary["rmsd"]), rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("CSV error metrics do not match the release log.")
    print("PASS complete-state budgets, release counts, RMSE and RMSD recomputation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    compile_parser = commands.add_parser("build", help="compile the replay and regression checks")
    compile_parser.add_argument("--native", action="store_true")
    commands.add_parser("check", help="check the core and recompute demo metrics")
    demo = commands.add_parser("demo", help="run a small three-method comparison")
    demo.add_argument("--output", default="results/demo")
    replay = commands.add_parser("run", help="replay one supplied experiment configuration")
    replay.add_argument("--config", required=True)
    replay.add_argument("--job")
    replay.add_argument("--input")
    replay.add_argument("--output", default="results/run")
    replay.add_argument("--trials", type=int)
    replay.add_argument("--seed", type=int)
    replay.add_argument("--methods", default="hfx,gmod_public,hx_phi")
    select = commands.add_parser("select", help="select a layout from public parameters")
    select.add_argument("--memory-kib", type=int, required=True)
    select.add_argument("--capacity", type=int, required=True)
    select.add_argument("--releases", type=int, required=True)
    select.add_argument("--epsilon", type=float, default=1)
    select.add_argument("--mode", choices=["full", "payload"], default="full")
    select.add_argument("--objective", choices=["controlled", "real_accuracy"], default="controlled")
    select.add_argument("--output", default="results/layout.json")
    args = parser.parse_args()
    if args.action == "build":
        build(args.native)
    elif args.action == "check":
        check()
    elif args.action == "demo":
        run(load_config(ROOT / "configs/demo.json"), args.output)
    elif args.action == "run":
        job = load_config(args.config, args.job)
        if args.trials is not None:
            job["trials"] = args.trials
        if args.seed is not None:
            job["seed"] = args.seed
        run(job, args.output, args.methods, args.input)
    else:
        try:
            from tools.layout import select_layout
        except ImportError as error:
            raise ValueError("Layout selection requires 'python -m pip install -r requirements.txt'.") from error
        profile = select_layout(args.memory_kib, args.capacity, args.releases,
                                args.epsilon, args.mode, args.objective)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(f"Layout: {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
