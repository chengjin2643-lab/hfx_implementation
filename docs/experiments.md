# 🧪 Running HFX experiments

[← Back to HFX](../README.md)

The experiment driver runs HFX, centrally private GMod, and bitmap-only Half-Xor
on the same endpoint stream and release schedule. Build the programs from the
repository directory before running a configuration:

```shell
python run_experiments.py build
python run_experiments.py check
python run_experiments.py demo
```

## ⚙️ Commands and parameters

### Replay an experiment

```shell
python run_experiments.py run --config configs/active32.json --trials 1 --output results/active32
```

| Option for `run` | Meaning |
| --- | --- |
| `--config` | Experiment JSON file, or the configuration index `configs/paper.json` |
| `--job` | Job key when selecting an entry from the configuration index |
| `--input` | Processed endpoint stream for a real-data experiment |
| `--methods` | Comma-separated selection of `hfx`, `gmod_public`, and `hx_phi` |
| `--trials` | Number of paired trials, overriding the configuration |
| `--seed` | Base seed for paired hashing and noise |
| `--output` | Output prefix, such as `results/active32` |

Workload, memory, and privacy settings come from the experiment JSON. The
[configuration guide](../configs/README.md) describes the 154 named jobs in
[`configs/paper.json`](../configs/paper.json).

### Select a public layout

```shell
python run_experiments.py select --memory-kib 32 --capacity 2000000 --releases 200 --mode full --objective controlled --output results/layout.json
```

| Option for `select` | Meaning |
| --- | --- |
| `--memory-kib` | Memory budget in KiB |
| `--capacity` | Declared upper bound on active cardinality, `Nmax` |
| `--releases` | Number of scheduled releases, `K` |
| `--epsilon` | Total privacy budget ε, default `1` |
| `--mode` | `full` for complete state, or `payload` for bitmap storage |
| `--objective` | `controlled` for the minimax rule, or `real_accuracy` for the density-aware rule |
| `--output` | Destination for the selected layout JSON |

Selection uses public parameters before processing. The generated JSON supplies
the `layout` member of an experiment configuration. A complete configuration also
specifies the workload and release schedule, as in
[`configs/demo.json`](../configs/demo.json). Keep its capacity, release count,
memory mode, and privacy parameters consistent with the selected layout.
The supplied selector and configurations use δ = `1e-6`.

## 💾 Memory and configuration

`payload` constrains the bitmap size. `full` constrains complete state, including
the mechanism object, auxiliary arrays, and decoder workspace. The replay checks
the complete-state limit for each selected method in `full` mode. In both modes,
`state_bytes` reports the measured mechanism state in bytes.

| JSON field | Meaning |
| --- | --- |
| `N`, `T`, `Nmax` | Synthetic active count, endpoint count, and declared capacity |
| `K`, `B`, `schedule` | Release count, endpoint interval, and release policy |
| `memory_kib`, `mode` | Memory budget and accounting mode |
| `epsilon`, `delta` | Privacy budget for the complete release sequence |
| `seed`, `first`, `trials` | Base seed and trial range |
| `pattern`, `completion`, `hot` | Synthetic workload settings |
| `layout` | Fixed column capacities, routing parameter, baseline geometry, and profile ID |

## 📄 Reading the outputs

For an output prefix `results/active32`, the driver writes:

- `active32_summary.csv`: one row per method and trial.
- `active32_releases.csv`: one row per method, trial, and release.
- `active32_job.tsv`: the concrete configuration passed to the C++ replay.

| Summary field | Meaning |
| --- | --- |
| `rmse` | Root mean square counting error across releases, in active-session units |
| `rmsd` | Root mean square relative error over releases with positive true counts |
| `service_rmse` | Error of the held estimate, weighted over endpoint slots |
| `state_bytes`, `w` | Measured complete state and sketch column count |
| `sigma` | Gaussian noise standard deviation |
| `release_count`, `processed` | Number of releases and processed endpoints |
| `clipped` | Number of releases whose noisy statistic was outside the decoding range |
| `hash_seed`, `noise_seed`, `profile_id` | Seeds and layout identifier for the trial |

**CSV `rmsd` values are fractions.** Multiply by 100 to express them as a
percentage. For example, `0.0068` corresponds to `0.68%` RMSD. `rmse` includes
zero-cardinality releases, while relative error is evaluated where the true count
is positive.

The release log records `slot`, `release_index`, `truth`, `estimate`,
`clean_estimate`, `clipped`, and `query`. `estimate` is the private count released
at that slot and held until the next release. Ground truth, clean estimates,
and queries provide evaluation diagnostics. Summary fields `clean_rmse` and
`clean_rmsd` measure the clean estimator for the same trial.

<a id="endpoint-format"></a>

## 📊 Real endpoint streams

Use a processed input stream with its matching configuration:

```shell
python run_experiments.py run --config configs/azure32.json --input data/azure_vm_endpoints_stable.bin --trials 1 --output results/azure32
```

The input uses the little-endian `HFXSTRM1` format, with no padding between fields:

| Section | Type and size | Content |
| --- | --- | --- |
| Magic | 8 bytes | ASCII `HFXSTRM1` |
| Count | `uint64`, 8 bytes | Number of endpoint records, equal to configuration `T` |
| Each record | `uint64`, 8 bytes | Session identifier |
| Each record | `int8`, 1 byte | `+1` for a start or `-1` for an end, used for evaluation truth |

Each endpoint record is nine bytes, so a stream with `T` records occupies
`16 + 9 × T` bytes. The evaluator uses the sign to track the true active count.
The three sketch methods receive only the identifier. In HFX, each occurrence
toggles its assigned bit.
