# Post-Quantum IoT Cryptography Benchmark

## 1. Introduction

The transition to Post-Quantum (PQ) computing necessitates a shift toward quantum-agile cryptographic designs. However, standard Post-Quantum Cryptography (PQC) algorithms often impose a significant "Security Tax"—excessive memory, latency, and energy overheads—rendering them impractical for resource-constrained IoT devices.

This project addresses this critical gap by conducting a comparative analysis of hybrid cryptographic architectures against a standalone quantum-resistant baseline. By integrating lightweight Authenticated Encryption (Ascon-128a) with various Key Encapsulation Mechanisms (KEMs), we evaluate the trade-offs between architectural complexity, performance overhead, and statistical robustness. The goal is to identify viable, efficient migration paths for 8-bit and 32-bit microcontrollers in a post-quantum landscape.

---

## 2. Cryptographic Models

This study benchmarks four distinct cryptographic architectures using a "KEM + DEM" (Key Encapsulation Mechanism + Data Encapsulation Mechanism) framework to simulate secure IoT sessions.

### Hybrid Architectures (KEM + Ascon-128a)

These models combine high-speed asymmetric key exchange with the efficiency of the Ascon-128a sponge construction.

* **Lattice-Based Hybrid (Model A):** **Kyber-512 + Ascon-128a**
* Utilizes ML-KEM (FIPS 203), the primary NIST standard for post-quantum key establishment.


* **Code-Based Hybrid:** **BIKE-L1 + Ascon-128a**
* Leverages the hardness of syndrome decoding; included to ensure algorithmic diversity and test trade-offs regarding key sizes and bandwidth.


* **Classical Hybrid (Baseline):** **X25519 + Ascon-128a**
* Uses Elliptic Curve Diffie-Hellman (ECC) to quantify the specific latency and memory costs of migrating from current pre-quantum standards.



### Standalone Baseline

* **Symmetric Quantum-Hardened:** **Ascon-80pq**
* A non-hybrid alternative mode of Ascon with an extended 160-bit key. It resists quantum key search attacks (Grover’s algorithm) without the overhead of asymmetric KEM operations, serving as a control variable for the "Security Tax" analysis.



---

## 3. Evaluation Metrics

To quantify performance and security robustness, all models were subjected to a dataset of 40,000 observations (10,000 per model) and assessed against the following metrics:

### Performance Overhead

* **Cryptographic Latency ():** The total computational time required to complete the encryption/decryption cycle.
* **Peak Memory Usage ():** The maximum RAM footprint utilized during operation, critical for constrained edge devices.

### Statistical Randomness

* **Shannon Entropy:** Measured in bits/byte (target ), ensuring the ciphertext is indistinguishable from random noise.
* **Serial Correlation Coefficient (SCC):** A measurement of the dependence between successive bytes in the ciphertext to detect potential patterns or biases in the hybrid encapsulation process.

---

## 4. How to run

```
make clean && make
./build_c
```

```
Applicable parameters/arguments:
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
  --single-thread {full|partial}
                        full: no worker threads in Phase 1 or Phase 2
                        partial: Phase 1 sequential, Phase 2 parallel (default)
  --model N             Run exactly one model:
                        1=Standalone_Ascon_80pq
                        2=Standalone_BIKE_L1
                        3=Standalone_Kyber512
                        4=Standalone_FrodoKEM_640_AES
                        5=Standalone_HQC_128
                        6=Standalone_ClassicMcEliece_348864
                        7=Standalone_X25519
                        8=Hybrid_ClassicMcEliece_348864_Ascon128a
                        9=Hybrid_FrodoKEM_640_AES_Ascon128a
                        10=Hybrid_HQC_128_Ascon128a
```
