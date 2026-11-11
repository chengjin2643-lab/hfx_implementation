# HFX

**Accurate and Memory-Efficient Private Continual Cardinality Estimation for Active Sets**

HFX estimates the number of active sessions using a fixed-memory reversible
sketch and differentially private releases.

## HFX overview

HFX processes session endpoints using their identifiers, without requiring
start or end labels. The two endpoints of a completed session toggle the same
bit and cancel, allowing the sketch to represent current activity without
storing a record for every session.

The paper introduces a nonuniform memory layout and an adaptive weighted
estimator. The layout is selected before processing and remains fixed, while
column weights adapt to the previous private estimate. At scheduled releases,
HFX adds Gaussian noise to the weighted bit counts and converts the result
into an active-count estimate. The noise is calibrated to protect the full
sequence of released counts, and the latest estimate is held between releases.

## Implementation

The benchmark drivers in this repository implement the coordinate-load-capped
variant with Gaussian binary-tree releases on canonicalized insertion/deletion
streams. The nonuniform layout and adaptive weighted estimator described in
the current paper are not implemented in these drivers.

- [experiment1_active_set.cpp](experiment1_active_set.cpp): C++20 implementation
  with packed sketch state, synthetic stream generation, and baseline methods.
- [experiment1_active_set.py](experiment1_active_set.py): NumPy reference
  implementation and experiment driver.

## Requirements

- C++20 compiler for the C++ driver. The commands below use MinGW-w64 `g++`
  on Windows, available on `PATH`.
- Python 3.10+ and NumPy for the Python driver.

```powershell
python -m pip install numpy
```

## Build

From the repository directory in PowerShell:

```powershell
g++ -std=c++20 -O3 -o experiment1_active_set.exe experiment1_active_set.cpp
```

Add `-march=native` to optimize for the machine running the experiment.

## Quick start

Run HFX on a synthetic stream of 20,000 updates with an 8 KiB sketch-state
budget and privacy parameters `epsilon = 1`, `delta = 1e-6`.

C++:

```powershell
.\experiment1_active_set.exe --T 20000 --memory-kb 8 --epsilon 1 --delta 1e-6 `
  --trials 1 --methods hfx --q 8 --hfx-noise-mode load_aware `
  --eval-every 1000 --output example_hfx_cpp.csv
```

Python:

```powershell
python .\experiment1_active_set.py --T 20000 --memory-kb 8 --epsilon 1 --delta 1e-6 `
  --trials 1 --methods hfx --q 8 --hfx-noise-mode load_aware `
  --eval-every 1000 --output example_hfx_python.csv
```

Both examples evaluate every 1,000 updates. Each output CSV records the trial
configuration, realized seeds, error metrics, and clipping statistics.

Use `--seed-stream`, `--seed-hash`, and `--seed-noise` to set the random seeds
independently for stream generation, hashing, and Gaussian noise.

## Baselines

The drivers include Flip-DP-Hash (`flipdp`), FM-Diff (`fm`), LL-Diff (`ll`),
and HLL-Diff (`hll`). Set `--methods hfx,flipdp,fm,ll,hll` to compare all five
methods under the configured sketch-state budget and evaluation schedule.

## License

[MIT](LICENSE).
