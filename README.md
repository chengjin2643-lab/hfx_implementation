# HFX: Private Continual Cardinality over Dynamic Active Sets

This repository contains the core implementation of HFX for continual
active-set cardinality estimation under event-level differential privacy.

## HFX overview

HFX expects a stream canonicalized upstream into active-set transitions. An
insertion activates an absent identifier, a deletion deactivates a present
identifier, and malformed endpoints are converted to no-op updates. HFX itself
keeps no per-identifier dictionary.

The released cardinality trajectory satisfies event-level continual
differential privacy, with one stream update as the privacy unit.

For public counter width `q`, let

```text
R = 2^q - 1
H = ceil(log2(T + 1))
rho = (sqrt(log(1/delta) + epsilon) - sqrt(log(1/delta)))^2
```

The Gaussian tree uses the following per-node variance:

```text
sigma^2 = 2 (R + 1) H / rho.
```

The default `--hfx-noise-mode load_aware` uses this variance.

The implementation maintains the active noisy-prefix sum incrementally, giving
O(1) release time and amortized O(1) node merges. HFX counters are bit-packed at
exactly `q` bits per coordinate. The reported sketch-state budget includes the
XOR matrix, packed load counters, and the two active binary-tree arrays.

## Implementation

- `experiment1_active_set.cpp` contains the C++20 implementation of HFX, the
  experiment driver, and the memory-matched baseline methods.
- `experiment1_active_set.py` provides a Python implementation with the same
  core workflow for quick experiments.

## Requirements

- A C++20 compiler. The commands below have been tested with MinGW-w64 `g++`
  on Windows.
- Python 3.10+ with NumPy for the Python implementation.

## Build

Compile the C++ implementation on Windows PowerShell:

```powershell
g++ -std=c++20 -O3 -march=native -o experiment1_active_set.exe experiment1_active_set.cpp
```

For a portable build, omit `-march=native`.

## Quick start

C++:

```powershell
.\experiment1_active_set.exe --T 20000 --q 8 `
  --hfx-noise-mode load_aware --output example_hfx_cpp.csv
```

Python:

```powershell
python .\experiment1_active_set.py --T 20000 --memory-kb 8 --epsilon 1 `
  --trials 1 --methods hfx --q 8 --hfx-noise-mode load_aware `
  --output example_hfx_python.csv
```

The implementation separates stream, public-hash, and Gaussian-noise seeds:

```text
--seed-stream 4801289
--seed-hash   63242691
--seed-noise  981723641
```

Each CSV records the realized seeds and configuration fields, including `rho`,
`q`, `R`, `sigma_node`, `clipping_rate`, and `eval_every`.

## Baselines

The experiment driver also provides Flip-DP-Hash, FM-Diff, LL-Diff, and
HLL-Diff. All methods use the configured sketch-state budget and evaluation
schedule.

## License

See `LICENSE`.
