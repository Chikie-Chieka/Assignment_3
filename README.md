# Post-Quantum Cryptography Benchmark — Artifact

This repo is the artifact for **<PAPER TITLE>** (<VENUE>, <YEAR>) and reproduces **Fig. <...>** and **Table <...>**.

## Repository layout
- `src/`, `include/`: C benchmark sources
- `bench_c_arm`: AArch64 binary for gem5 (SE mode)
- `experiment_final.py`: full experiment driver
- `analysis.py` + `requirements.txt`: plot/table regeneration
- `saved_output/`: reference outputs (golden)
- `analysis_output/`: generated plots/tables
- `gem5_tests/`: gem5 helpers/configs
- `oqs_arm/`: PQC dependency build for ARM

## Prerequisites (Arch Linux x86_64 baseline)
Hardware baseline: laptop (x86_64), **2 cores pinned**, **1 GiB RAM cap** (via `systemd-run`) for “constrained” runs.  
Packages (minimum): `base-devel cmake python python-venv git`  
Optional (gem5 build): `scons protobuf boost capstone` (+ gem5’s documented deps)

## Build (native x86_64)
```bash
make clean && make        # produces ./bench_c
./bench_c --help          # lists models + flags
```

## bench_c parameters
```
./bench_c -help
usage: ./bench_c [-h] [--iterations ITERATIONS] [--payload-bytes PAYLOAD_BYTES]
          [--aad AAD] [--seed SEED]
          [--ent-payload-mb ENT_PAYLOAD_MB] [--skip-latency] [--skip-ent]
          [--ent-iterations ENT_ITERATIONS] [--no-csv | --output none]
          [--model N]

optional arguments:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        (default: 100)
  --payload-bytes PAYLOAD_BYTES
                        (default: 4096)
  --aad AAD             (default: )
  --seed SEED           (default: 1337)
  --ent-iterations ENT_ITERATIONS
                        (default: 50)
  --ent-payload-mb ENT_PAYLOAD_MB
                        ENT test payload size in MB (default: 1)
  --skip-latency        Skip latency/memory benchmarking phase (testing_process.csv + results.json)
  --skip-ent            Skip ENT randomness testing phase (ENT_Test.csv)
  --no-csv, --output none
                        Disable all CSV output files
  --single-thread {none|full|partial}
                        none: Phase 1 + Phase 2 use worker threads (default)
                        full: no worker threads in Phase 1 or Phase 2
                        partial: Phase 1 sequential, Phase 2 parallel
  --model N             Run exactly one model:
                        1=Standalone_Ascon_80pq
                        2=Standalone_BIKE_L1
                        3=Standalone_Kyber512
                        4=Standalone_FrodoKEM_640_AES
                        5=Standalone_ClassicMcEliece_348864
                        6=Standalone_X25519
                        7=Hybrid_BIKE_L1_Ascon128a
                        8=Hybrid_Kyber512_Ascon128a
                        9=Hybrid_FrodoKEM_640_AES_Ascon128a
                        10=Hybrid_ClassicMcEliece_348864_Ascon128a
                        11=Hybrid_X25519_Ascon128a
```

## Run: 5-minute smoke test (native)
Runs a single fast model end-to-end (including ENT) to verify the toolchain.
```bash
sudo systemd-run --scope -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=200% bash -lc 'cd "$PWD" && taskset -c 0,1 ./bench_c --model 8 --payload-bytes 4096 --iterations 200 --ent-payload-mb 1 --ent-iterations 50 --seed 67 --single-thread none'
```

## Run: full experiment (native, paper settings)
```bash
sudo systemd-run --scope -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=200% bash -lc 'cd "$PWD" && taskset -c 0,1 ./bench_c --payload-bytes 65536 --iterations 10000 --ent-payload-mb 10 --ent-iterations 1000 --seed 67 --single-thread none'
```

## gem5 setup (ARMv8-A, SE mode)
Paper microarch config: `TimingSimpleCPU` @2GHz; L1D 64KB/2-way, L1I 32KB/2-way, L2 2MB/8-way; DRAM 4GB; **single-thread**; **1 warmup + 50 measured**.

1) Build gem5 (host x86_64):
```bash
git clone https://github.com/gem5/gem5 && cd gem5
scons build/ARM/gem5.opt -j"$(nproc)"
```

2) Run one model (example: model 8 = Hybrid_Kyber512_Ascon128a):
```bash
GEM5=</path/to/gem5>
OUT=gem5_out/model8
"$GEM5"/build/ARM/gem5.opt --outdir="$OUT" "$GEM5"/configs/example/se.py \
  --cpu-type=TimingSimpleCPU --cpu-clock=2GHz --caches --l2cache \
  --l1d_size=64kB --l1i_size=32kB --l2_size=2MB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=8 \
  --mem-size=4GB --cmd="$PWD/bench_c_arm" \
  --options="--model 8 --payload-bytes 65536 --iterations 51 --seed 67 --skip-ent --single-thread full"
```

Extract cycles from `"$OUT"/stats.txt` (e.g., `system.cpu.numCycles`). Discard warmup; report median of last 50.

## Expected outputs
Native run (default): `testing_process.csv`, `results.json`, `ENT_Test.csv` (see `./bench_c --help` for `--skip-*` and `--no-csv`).  

gem5 run: `gem5_out/*/stats.txt`, `config.json`, `config.ini` (+ `simout`).

