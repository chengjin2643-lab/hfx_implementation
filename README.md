# Dynamic Differentially Private Cardinality Estimation under Continual Release





## Preface



This repository **HFX** implements the **HFX** algorithm and baseline methods for the *Dynamic Differentially Private Cardinality Continual Releases (D-DPCCR)* problem.



## Implementation of our algorithms and baselines



| **Method**     | **Reference**                                                |
| -------------- | ------------------------------------------------------------ |
| **HFX (Ours)** | [hfx.h](https://www.google.com/search?q=hfx/hfx.h)           |
| Flip-DP        | [nips_hfx.h](https://www.google.com/search?q=hfx/nips_hfx.h) |
| FM-Turnstile   | [fm_count.h](https://www.google.com/search?q=hfx/fm_count.h) |
| LL-Turnstile   | [ll_count.h](https://www.google.com/search?q=hfx/ll_count.h) |
| HLL-Turnstile  | [hll_count.h](https://www.google.com/search?q=hfx/hll_count.h) |



## Environment



- **OS**: Ubuntu 18.04 / 20.04 LTS
- **Compiler**: g++ (Supports C++20, e.g., gcc version 10.5.0 or higher)



## Compile and run



Download this repository, unzip it, and run the following commands in the source directory.



### 1. Main Experiment (Memory Comparison)



This reproduces the insertion-only comparison experiments.

Bash

```
g++ -std=c++20 -O3 -o dpccr dpccr.cpp MurmurHash3.cpp parameters.cpp
./dpccr
```



### 2. Deletion Experiment (Dynamic Stream)



This reproduces the experiments with element deletions.

Bash

```
g++ -std=c++20 -O3 -o delete_exp delete.cpp MurmurHash3.cpp parameters.cpp
./delete_exp
```



