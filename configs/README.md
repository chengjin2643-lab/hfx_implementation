# ⚙️ Experiment configurations

[← Back to HFX](../README.md) · [Experiment guide](../docs/experiments.md)

The JSON files specify workloads, privacy parameters, release schedules, and
layouts for the C++ benchmark driver.

| File | Workload | Memory setting |
| --- | --- | --- |
| `demo.json` | 1,000 active sessions, 21,000 endpoints, 200 releases, one trial | 32 KiB complete state |
| `active32.json` | 1,000,000 active sessions, 3,000,000 endpoints, 200 releases | 32 KiB complete state |
| `azure32.json` | Azure VM endpoint stream, 3,920,600 endpoints, 393 releases | 32 KiB payload |
| `paper.json` | 154 configurations covering real streams, privacy budgets, active scale, stream history, repeated activity, and release intervals | Per-configuration |

## Running a configuration

Build the programs before running an experiment:

```powershell
python run_experiments.py build
python run_experiments.py check
python run_experiments.py demo
```

Run the million-session configuration for one trial:

```powershell
python run_experiments.py run --config configs/active32.json --trials 1 --output results/active32
```

Select a configuration from `paper.json` using its job key:

```powershell
python run_experiments.py run --config configs/paper.json --job "active/n1000000_t3000000/m32/e1" --trials 1 --output results/active32
```

For a real-data configuration, provide the processed endpoint stream:

```powershell
python run_experiments.py run --config configs/azure32.json --input data/azure_vm_endpoints_stable.bin --output results/azure32
```

The same Azure configuration is available as
`real/azure-vm/m32/e1` in `paper.json`. Its `input_filename` identifies the
expected processed file. All configurations use `delta = 1e-6`.

## Fields

- `mode`: `full` limits complete state, while `payload` limits bitmap storage.
  `theory` uses fixed weights computed at `Nmax / 2`, with a payload budget.
- `N`, `T`, `Nmax`: synthetic active count, endpoint count, and declared capacity.
- `K`, `B`, `schedule`: release count, endpoint interval, and release policy.
- `memory_kib`, `epsilon`, `delta`: memory and privacy budgets.
- `seed`, `first`, `trials`: reproducible seed and trial range.
- `pattern`, `completion`, `hot`: synthetic stream parameters.
- `layout`: routing parameter `q16`, per-column bit capacities `rows`, GMod width
  and rows, and the profile identifier. Layouts stay fixed during a replay.

Real-data entries have `input: null` and take the input path from `--input`.
The real-data and privacy-budget entries use the same density-aware layout
selection rule. The other entries retain their respective experimental layouts.
