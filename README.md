# HFX

**Accurate and Memory-Efficient Private Continual Cardinality Estimation for Active Sets**

HFX estimates the number of active sessions using a fixed-memory reversible
sketch and differentially private releases.

## HFX overview

HFX processes session endpoints using their identifiers, without requiring
start or end labels. The two endpoints of a completed session toggle the same
bit and cancel, allowing the sketch to represent current activity without
storing a record for every session.

A nonuniform memory layout is selected before processing and remains fixed.
Column weights adapt to the previous private estimate. At scheduled releases,
HFX adds Gaussian noise to the weighted bit counts and converts the result
into an active-count estimate. The noise is calibrated to protect the full
sequence of released counts, and the latest estimate is held between releases.

## Implementation

- [src/lean_hfx.hpp](src/lean_hfx.hpp): reversible updates, bounded adaptive
  weights, Gaussian releases, and cardinality decoding.
- [tools/layout](tools/layout): layout selection using the declared memory
  budget, capacity bound, release count, and privacy parameters.
- [benchmarks/replay.cpp](benchmarks/replay.cpp): paired experiments for HFX,
  GMod, and Half-Xor on synthetic or processed endpoint streams.
- [configs](configs): the paper's experiment configurations and a small example.
- [run_experiments.py](run_experiments.py): build, check, and experiment commands.

## Requirements

- A 64-bit GCC or Clang compiler with C++20 and `unsigned __int128` support.
  The Windows build is tested with MinGW-w64 GCC.
- Python 3.10+ for the experiment commands.
- NumPy, pandas, and SciPy for selecting new layouts:

```shell
python -m pip install -r requirements.txt
```

The compiler must be on `PATH`. Use `CXX` to select a compiler or
`HFX_MINGW_BIN` to specify the MinGW-w64 directory on Windows.

## Quick start

Run these commands from the repository directory:

```shell
python run_experiments.py build
python run_experiments.py check
python run_experiments.py demo
```

The example compares the three sketches over 21,000 endpoint updates with
200 releases and a 32 KiB complete-state budget. Results are written to
`results/demo_summary.csv` and `results/demo_releases.csv`.

The checks cover endpoint cancellation, bounded weights, decoding endpoints,
update-path equivalence, memory budgets, and error-metric recomputation.
Use `build --native` to enable machine-specific compiler optimizations.

## Experiments

Run the one-million-active-session configuration with one trial:

```shell
python run_experiments.py run --config configs/active32.json --trials 1 --output results/active32
```

For an Azure endpoint stream, use the corresponding configuration and input:

```shell
python run_experiments.py run --config configs/azure32.json --input data/azure_vm_endpoints_stable.bin --trials 1 --output results/azure32
```

[configs/README.md](configs/README.md) describes the full configuration index.
`--methods hfx,gmod_public,hx_phi` selects HFX, centrally private GMod with
publicly selected width, and the bitmap-only Half-Xor estimator. The methods
use the same input, release schedule, and privacy parameters. `--seed` sets
the base seed and `--trials` sets the number of paired trials.

Memory mode `payload` constrains the bitmap size. Mode `full` constrains
complete state, including auxiliary arrays and decoder workspace. The replay
checks the complete-state limit for every method in `full` mode and reports
actual state bytes in both modes.

The benchmark CSVs contain ground truth and clean estimates for evaluating
error, alongside private estimates. They are experiment logs rather than
private output streams.

### Layout selection

Select a layout using public parameters before processing a stream:

```shell
python run_experiments.py select --memory-kib 32 --capacity 2000000 --releases 200 --mode full --objective controlled --output results/layout.json
```

`controlled` uses the minimax rule for controlled workloads.
`real_accuracy` uses the density-aware rule for real-stream experiments.
Both rules use the declared parameters and public risk model. Layout selection
and the supplied configurations use `delta = 1e-6`.

### Endpoint format

Processed input uses the little-endian `HFXSTRM1` format: an eight-byte magic
string, a `uint64` endpoint count, then records containing a `uint64` session
identifier and an `int8` sign. The sign is used to compute evaluation truth.
The three sketch methods receive only the identifier. Each occurrence toggles
its assigned bit.

## License

[MIT](LICENSE).
