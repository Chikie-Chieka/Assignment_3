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

Would you like me to draft a Python snippet for calculating the Shannon Entropy or SCC for your dataset?