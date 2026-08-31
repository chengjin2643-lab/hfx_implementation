# HFX: Private Continual Cardinality over Dynamic Active Sets

This repository contains the core implementation of HFX for continual
active-set cardinality estimation under event-level differential privacy.

## Current implementation

- `experiment1_active_set.cpp` is the authoritative C++20 implementation used
  for paper-scale experiments. It includes HFX, the experiment driver, and the
  memory-matched reference methods.
- `experiment1_active_set.py` is a readable Python reference implementation for
  inspection and small smoke tests.

The older header-based files remain for compatibility with the initial public
prototype. New results should use `experiment1_active_set.cpp`.

Plotting scripts, generated figures, raw datasets, compiled binaries, and full
experiment outputs are intentionally not included in this core-code release.

## Model and calibration

HFX expects a stream canonicalized upstream into active-set transitions. An
insertion activates an absent identifier, a deletion deactivates a present
identifier, and malformed endpoints are converted to no-op updates. HFX itself
keeps no per-identifier dictionary.

The privacy guarantee is event-level continual differential privacy over the
released cardinality trajectory. It is not a user-level guarantee; multiple
events belonging to one person require a separate group-privacy or user-level
analysis.

For public counter width `q`, let

```text
R = 2^q - 1
H = ceil(log2(T + 1))
rho = (sqrt(log(1/delta) + epsilon) - sqrt(log(1/delta)))^2
```

The Gaussian tree uses the theorem-aligned per-node variance

```text
sigma^2 = 2 (R + 1) H / rho.
```

The default `--hfx-noise-mode load_aware` implements this calibration.
`unit_diagnostic` is retained only for inspecting historical runs and must not
be reported as a theorem-aligned private HFX result.

The implementation maintains the active noisy-prefix sum incrementally, giving
O(1) release time and amortized O(1) node merges. HFX counters are bit-packed at
exactly `q` bits per coordinate. The reported sketch-state budget includes the
XOR matrix, packed load counters, and the two active binary-tree arrays.

## Build

Requirements:

- a C++20 compiler (tested with MinGW-w64 `g++`);
- Python 3.10+ and NumPy for the reference implementation.

Compile the C++ implementation on Windows PowerShell:

```powershell
g++ -std=c++20 -O3 -march=native -o experiment1_active_set.exe experiment1_active_set.cpp
```

For a portable build, omit `-march=native`.

## Smoke test

C++:

```powershell
.\experiment1_active_set.exe --preset smoke --T 20000 --q 8 `
  --hfx-noise-mode load_aware --output smoke_load_aware.csv
```

Python:

```powershell
python .\experiment1_active_set.py --preset smoke --T 20000 --q 8 `
  --hfx-noise-mode load_aware --output smoke_python.csv
```

The implementation separates stream, public-hash, and Gaussian-noise seeds:

```text
--seed-stream 4801289
--seed-hash   63242691
--seed-noise  981723641
```

Each CSV records the realized seeds and calibration fields, including `rho`,
`q`, `R`, `sigma_node`, `clipping_rate`, and `eval_every`.

## References implemented by the driver

The experiment driver also provides Flip-DP-Hash, FM-Diff, LL-Diff, and
HLL-Diff reference methods. Flip-DP-Hash is a fixed-memory hashed adaptation and
does not inherit the guarantee of the original per-item-state construction.
The reference methods share the configured sketch-state budget and evaluation
schedule, but their privacy units and assumptions are not claimed to be
identical to HFX.

## License

See `LICENSE`.
