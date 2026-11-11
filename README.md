# HFX: Accurate and Memory-Efficient Private Continual Cardinality Estimation for Active Sets

![Language](https://img.shields.io/badge/Language-C%2B%2B20-255B85)
![Python](https://img.shields.io/badge/Python-3.10%2B-27665F)
[![License](https://img.shields.io/badge/License-MIT-54876B)](LICENSE)

How many sessions are active in a service right now? HFX tracks sessions as
they start and end using a compact sketch. It continually releases accurate
counts with differential privacy, while keeping memory use fixed.

- ⚡ **High accuracy**
- 🧠 **Low memory overhead**
- 🔒 **Privacy protection**

[🚀 Quickstart](#quickstart) · [🧪 Examples](#examples) ·
[🧩 Implementation](#implementation) · [📊 Workloads](#workloads) ·
[🎛️ Parameters](#parameters)

## Quickstart

You need Python 3.10+ and a 64-bit GCC or Clang compiler with C++20 and
`unsigned __int128` support. The demo generates its own input and uses the
C++ standard library.

```shell
git clone https://github.com/chengjin2643-lab/hfx_implementation.git
cd hfx_implementation
python run_experiments.py build
python run_experiments.py check
python run_experiments.py demo
```

The demo compares three methods on 21,000 endpoints with 200 releases and
a 32 KiB complete-state budget. Open `results/demo_summary.csv` for accuracy
and memory use, and `results/demo_releases.csv` for the individual releases.

<details>
<summary>Windows compiler setup</summary>

With MinGW-w64, put its `bin` directory on `PATH` or set it in PowerShell:

```powershell
$env:HFX_MINGW_BIN = "C:\msys64\mingw64\bin"
python run_experiments.py build
```

Set `CXX` to choose another compiler. Use `build --native` for
machine-specific optimizations.

</details>

## Examples

**One million active sessions.** Run one trial over 3,000,000 endpoints with
200 releases and a 32 KiB complete-state budget:

```shell
python run_experiments.py run --config configs/active32.json --trials 1 --output results/active32
```

**Repeated activity.** Test reactivated identifiers over 100,000 endpoints
with a 32 KiB bitmap:

```shell
python run_experiments.py run --config configs/paper.json --job reuse/reuse_t100000/m32/e1 --trials 1 --output results/repeated_activity
```

**HFX alone.** Select a single method with `--methods`:

```shell
python run_experiments.py run --config configs/active32.json --methods hfx --trials 1 --output results/hfx
```

Each command writes a summary CSV, a release CSV, and the replay configuration.
The [experiment guide](docs/experiments.md) explains the output columns.
CSV `rmsd` values are fractions, so `0.0068` means `0.68%` relative error.

## Implementation

HFX receives endpoint identifiers without requiring start or end labels.
The two endpoints of a completed session toggle the same bit and cancel.
Its memory layout stays fixed, while bounded weights adapt to the previous
private estimate. Gaussian noise protects the sequence of released counts.

The [comparison driver](benchmarks/replay.cpp) evaluates three sketches on
the same stream, release schedule, and privacy budget:

| Method | Command identifier | Description |
| --- | --- | --- |
| [**HFX**](src/lean_hfx.hpp) | `hfx` | Nonuniform memory layout and adaptive weighted estimation |
| [GMod](src/frozen/icde_audit.cpp) | `gmod_public` | Central-DP adaptation with publicly selected width |
| [Half-Xor](src/frozen/icde_audit.cpp) | `hx_phi` | Bitmap-only φ estimator with 32 columns |

## Workloads

| Workload | Activity | Source or configuration |
| --- | --- | --- |
| Synthetic | Active sessions, repeated activity, and long histories | [Configuration index](configs/README.md) |
| CitiBike | Active rides | [System data](https://citibikenyc.com/system-data) |
| TLC-Yellow | Active taxi trips | [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| UGR16 | Active network flows | [UGR'16 dataset](https://nesg.ugr.es/nesg-ugr16/index.php) |
| Azure VM | Virtual-machine lifecycles | [Azure Public Dataset V1](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md) |

For a processed Azure endpoint stream:

```shell
python run_experiments.py run --config configs/azure32.json --input data/azure_vm_endpoints_stable.bin --trials 1 --output results/azure32
```

The [input-format guide](docs/experiments.md#endpoint-format) describes the
`HFXSTRM1` binary format. Each retained interval supplies two occurrences of
an activity identifier.

## Parameters

| Option for `run` | Meaning |
| --- | --- |
| `--config` | Workload, privacy, and layout settings in a JSON file |
| `--job` | Named setting from `configs/paper.json` |
| `--methods` | Comma-separated `hfx,gmod_public,hx_phi` |
| `--trials`, `--seed` | Trial count and base seed |
| `--input` | Processed endpoint stream |
| `--output` | Output prefix |

Memory and privacy settings come from the selected configuration.
`payload` limits bitmap storage. `full` limits complete state, including
auxiliary arrays and decoder workspace.

To select a new layout from public parameters, install the Python dependencies
and run:

```shell
python -m pip install -r requirements.txt
python run_experiments.py select --memory-kib 32 --capacity 2000000 --releases 200 --mode full --objective controlled --output results/layout.json
```

The selected profile supplies the `layout` field in a workload configuration.
See the [experiment guide](docs/experiments.md) for all selection parameters
and the [configuration index](configs/README.md) for the 154 supplied settings.

## License

Released under the [MIT License](LICENSE).
