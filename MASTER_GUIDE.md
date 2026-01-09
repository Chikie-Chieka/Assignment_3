# MASTER_GUIDE: PQC Cryptography Benchmarking on Ubuntu/WSL2

**Last Updated:** January 2026  
**Scope:** `main_experiment.py` and `main_experiment_enhanced.py`  
**Target Platforms:** Ubuntu 20.04+, WSL2 (Windows 11), macOS (best-effort)

---

## Table of Contents

1. [Installation Steps](#installation-steps)
2. [Dependency Troubleshooting](#dependency-troubleshooting)
3. [Metrics Explanation](#metrics-explanation)
4. [Iterations Detail](#iterations-detail)
5. [Analysis Section](#analysis-section)
6. [.gitignore Template](#gitignore-template)
7. [Quick Reference](#quick-reference)

---

## Installation Steps

### Prerequisites

Before running benchmarks, ensure you have:
- **Python 3.8+** (recommended: 3.10+)
- **pip** (Python package manager)
- **git** (for version control)
- **Build tools** (gcc, make, etc.)
- **libssl-dev** (for crypto libraries)

### Ubuntu / WSL2 Installation (Recommended)

#### Step 1: Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential libssl-dev python3-dev python3-pip python3-venv git
```

**Explanation:**
- `build-essential`: Compiler toolchain for C/C++ dependencies
- `libssl-dev`: OpenSSL headers (required for many crypto libs)
- `python3-dev`: Python development files (headers)
- `python3-venv`: Virtual environment support

#### Step 2: Create a Virtual Environment

```bash
# Navigate to your project directory
cd /path/to/Assignment_3

# Create virtual environment named 'pqc_lab'
python3 -m venv pqc_lab

# Activate it
source pqc_lab/bin/activate  # Linux/WSL2
# OR on Windows PowerShell:
# pqc_lab\Scripts\Activate.ps1
```

**Why virtual environment?**
- Isolates dependencies (doesn't affect system Python)
- Prevents version conflicts
- Easy cleanup (just delete the directory)
- Reproducible across machines

**Verify activation:**
```bash
which python  # Should show /path/to/pqc_lab/bin/python
python --version
```

#### Step 3: Upgrade pip, setuptools, wheel

```bash
python -m pip install --upgrade pip setuptools wheel
```

**Expected output:**
```
Successfully installed pip-25.2 setuptools-70.0.0 wheel-0.42.0
```

#### Step 4: Install Core Dependencies

```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

**Packages:**
- `numpy`: Numerical arrays
- `scipy`: Statistical functions
- `pandas`: Data manipulation (CSV handling)
- `matplotlib`: Plotting/visualization
- `scikit-learn`: Machine learning (optional for advanced analysis)

**Expected time:** 2-5 minutes

#### Step 5: Install liboqs-python (Post-Quantum Cryptography)

This is the **most critical** dependency. It provides Kyber512 and BIKE-L1 implementations.

##### Option A: Install from PyPI (Easiest)

```bash
pip install oqs
```

**Expected output:**
```
Collecting oqs
Downloading oqs-0.14.1-cp313-cp313-linux_x86_64.whl
Installing collected packages: oqs
Successfully installed oqs-0.14.1
```

**Verify installation:**
```bash
python -c "import oqs; print(oqs.Kem.get_supported()[:3])"
```

**Expected output (first 3):**
```
['Kyber512', 'Kyber768', 'Kyber1024', ...]
```

##### Option B: Build from Source (Ubuntu/WSL2 - If PyPI fails)

If PyPI wheel doesn't work for your platform:

```bash
# 1. Install build dependencies
sudo apt install -y liboqs0 liboqs-dev

# 2. Clone liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j4
sudo make install

# 3. Build Python bindings
cd ../../liboqs-python
pip install .
```

**Verify:**
```bash
python -c "import oqs; print('Kyber512' in oqs.Kem.get_supported())"
```

**Expected output:** `True`

#### Step 6: Install PyNaCl (X25519/Classical ECDH)

```bash
pip install PyNaCl
```

**Verify:**
```bash
python -c "from nacl.public import PrivateKey; print('PyNaCl OK')"
```

**Expected output:** `PyNaCl OK`

#### Step 7: Install Ascon (AEAD Cipher)

This is the symmetric cipher used in all models' DEM phase.

```bash
pip install ascon
```

**Verify:**
```bash
python -c "from ascon import Ascon; print('Ascon OK')"
```

**Expected output:** `Ascon OK`

#### Step 8: Install Memory Profiling (Optional but Recommended)

For analyzing memory usage during benchmarks:

```bash
pip install memory-profiler psutil
```

#### Step 9: Verify Complete Installation

```bash
# Run the verification script
python -c "
import sys
import oqs
from nacl.public import PrivateKey
from ascon import Ascon

print('✓ Python:', sys.version.split()[0])
print('✓ oqs (liboqs):', hasattr(oqs, 'Kem'))
print('✓ PyNaCl:', PrivateKey is not None)
print('✓ Ascon:', Ascon is not None)
print()
print('All dependencies installed!')
"
```

**Expected output:**
```
✓ Python: 3.10.12
✓ oqs (liboqs): True
✓ PyNaCl: True
✓ Ascon: True

All dependencies installed!
```

### Windows (Direct - NOT Recommended)

While WSL2 is recommended, native Windows installation:

1. **Install Python 3.10+** from python.org
2. **Install Visual C++ Build Tools** (Microsoft)
3. **Install OpenSSL** (from https://www.openssl.org/community/binaries.html)
4. **Set environment variables:**
   ```powershell
   $env:OPENSSL_DIR = "C:\OpenSSL-Win64"
   ```
5. **Follow Steps 2-9 above** in PowerShell

**Note:** Native Windows often fails at liboqs compilation. WSL2 is strongly preferred.

### macOS Installation (Best-Effort)

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 openssl liboqs

# Create venv and install Python packages (same as Ubuntu)
python3 -m venv pqc_lab
source pqc_lab/bin/activate
pip install --upgrade pip setuptools wheel
pip install oqs PyNaCl ascon numpy scipy pandas matplotlib

# May need to link OpenSSL:
export LDFLAGS="-L$(brew --prefix openssl)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include"
pip install oqs --no-binary :all:  # Force build from source if needed
```

**Known Issue:** liboqs on macOS may require building from source. See "Dependency Troubleshooting" if needed.

---

## Dependency Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'oqs'`

**Error Message:**
```
Traceback (most recent call last):
  File "main_experiment.py", line 45, in _import_liboqs
    import oqs
ModuleNotFoundError: No module named 'oqs'
```

**Root Causes:**
1. liboqs-python not installed
2. Wrong virtual environment activated
3. pip install failed silently

**Solutions:**

**Solution 1a: Direct reinstall**
```bash
# Deactivate and reactivate venv to clear any caches
deactivate
source pqc_lab/bin/activate
pip uninstall oqs -y
pip install oqs --no-cache-dir
```

**Solution 1b: Verify pip is correct**
```bash
which pip  # Should show pqc_lab/bin/pip
pip list | grep oqs  # Should list oqs
```

**Solution 1c: Check Python path**
```bash
python -c "import sys; print(sys.executable)"
# Should show /path/to/pqc_lab/bin/python
python -c "import site; print(site.getsitepackages())"
# Should show /path/to/pqc_lab/lib/python3.x/site-packages
```

**Solution 1d: Build from source (Ubuntu/WSL2)**
```bash
sudo apt install -y liboqs-dev
pip install oqs --no-binary :all: --force-reinstall
```

---

### Issue 2: `ImportError: liboqs.so.0: cannot open shared object file`

**Error Message:**
```
ImportError: liboqs.so.0: cannot open shared object file: No such file or directory
```

**Root Cause:** The compiled liboqs library is not in the system's library search path.

**Solutions:**

**Solution 2a: Install system liboqs (Ubuntu/WSL2)**
```bash
sudo apt install -y liboqs0  # Runtime library
sudo ldconfig  # Update library cache
pip install oqs --force-reinstall
```

**Solution 2b: Set LD_LIBRARY_PATH (if custom build)**
```bash
# If you built liboqs from source:
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
python main_experiment.py --iterations=10
```

**Solution 2c: Reinstall with explicit linking**
```bash
sudo apt install -y liboqs-dev
pip install oqs --no-binary :all: --force-reinstall
```

---

### Issue 3: Version Mismatch: `oqs` vs `liboqs-python`

**Error Message:**
```
oqs.OQSException: Unsupported KEM: Kyber512
```

**Root Cause:** liboqs-python is built against an older/newer liboqs version that doesn't support Kyber512.

**Diagnostic:**
```bash
# Check what KEMs are available
python -c "import oqs; print(oqs.Kem.get_supported())"
```

**Example (outdated build):**
```
Available KEMs: ['MLKEM512', 'MLKEM768', 'MLKEM1024']  # NIST renamed Kyber → ML-KEM
# But our code expects 'Kyber512' (old name)
```

**Solutions:**

**Solution 3a: Update to latest (RECOMMENDED)**
```bash
pip install --upgrade oqs
```

**Solution 3b: Use NIST names (if using newer liboqs)**
Edit `main_experiment.py` line where models are defined:
```python
# OLD (liboqs 0.13 and earlier)
kem = oqs.Kem("Kyber512")

# NEW (liboqs 0.14+, NIST names)
kem = oqs.Kem("ML-KEM-512")
```

**Solution 3c: Check version and pin it**
```bash
# See installed version
pip show oqs | grep Version

# Pin a known-good version
pip install 'oqs==0.14.1'
```

**Reference Version Map:**
| liboqs | Kyber Name | Status |
|--------|-----------|--------|
| 0.13 | Kyber512 | Deprecated |
| 0.14+ | ML-KEM-512 | NIST standard |

---

### Issue 4: Ascon API Mismatch

**Error Message:**
```
AttributeError: module 'ascon' has no attribute 'Ascon'
```

**Root Cause:** Multiple Ascon packages exist on PyPI with different APIs:
1. `ascon` by `dsprenkels/ascon` (simple API)
2. `ascon` fork with different interface
3. System-installed ascon (AEAD lib, C extension)

**Diagnostic:**
```bash
python -c "import ascon; print(dir(ascon))" | head -20
```

**Example outputs:**

**Good (Expected):**
```
['Ascon', 'AeadCipher', 'AsconPermutation', ...]
```

**Bad (Wrong package):**
```
['encrypt', 'decrypt', ...]  # Different API structure
```

**Solutions:**

**Solution 4a: Install correct package**
```bash
pip uninstall ascon -y
pip install ascon==0.2.0  # Known-good version
```

**Solution 4b: Verify API at runtime**
Add this check to your script:
```python
import ascon
if hasattr(ascon, 'Ascon'):
    print("✓ Correct ascon package")
else:
    print("✗ Wrong ascon package - uninstall and reinstall")
    sys.exit(1)
```

**Solution 4c: Use specific import**
```python
# Try importing from the package in various ways
try:
    from ascon import Ascon
except ImportError:
    try:
        from ascon.ascon import Ascon
    except ImportError:
        raise RuntimeError("Cannot find Ascon implementation")
```

**Solution 4d: Pin the version (recommended)**
Create `requirements.txt`:
```
oqs==0.14.1
PyNaCl==1.6.2
ascon==0.2.0
numpy==1.24.0
scipy==1.11.0
pandas==2.0.0
matplotlib==3.7.0
```

Then install:
```bash
pip install -r requirements.txt
```

---

### Issue 5: `perf_counter_ns()` Not Available

**Error Message:**
```
AttributeError: 'module' object has no attribute 'perf_counter_ns'
```

**Root Cause:** Python < 3.7 (perf_counter_ns was added in 3.7)

**Solution:**
```bash
python --version  # Check current version
# If < 3.7:
python3 -m venv pqc_lab --python=/usr/bin/python3.10  # Upgrade to 3.10+
source pqc_lab/bin/activate
pip install --upgrade pip
```

---

### Issue 6: Memory/File Limits

**Error Message (for enhanced version with large iterations):**
```
MemoryError: Unable to allocate X MB
OSError: [Errno 24] Too many open files
```

**Root Cause:** CSV writing on 10,000+ iterations exhausts resources

**Solutions:**

**Solution 6a: Increase file descriptor limit**
```bash
# Check current limit
ulimit -n

# Increase to 4096 (temporary)
ulimit -n 4096

# Make permanent (Linux) in ~/.bashrc:
echo 'ulimit -n 4096' >> ~/.bashrc
source ~/.bashrc
```

**Solution 6b: Use smaller iterations for testing**
```bash
python main_experiment_enhanced.py --iterations=100  # Not 10000
```

**Solution 6c: Increase swap (WSL2)**
In `/etc/wsl.conf`:
```ini
[interop]
appendWindowsPath = true

[wsl2]
memory=8GB
swap=4GB
```

Then restart WSL:
```powershell
wsl.exe --shutdown
wsl.exe  # Restart
```

---

## Metrics Explanation

### Overview

Two entry points produce different output formats:

- **`main_experiment.py`**: JSON-only (aggregated statistics)
- **`main_experiment_enhanced.py`**: JSON + CSV (per-iteration + aggregated)

### JSON Output (`results.json`)

#### Structure

```json
{
  "meta": {
    "iterations": 100,
    "payload_bytes": 256,
    "aad_bytes": 0,
    "seed": 1337,
    "csv_output": "latency_results.csv",
    "notes": [
      "Models A-C: KEM/DH -> HKDF-SHA256 -> Ascon-128a",
      "Model D: PSK -> Ascon-80pq (no KEM, no HKDF)"
    ],
    "versions": {
      "python": "3.10.12",
      "platform": "Linux-5.15.146.1-microsoft-standard-x86_64-with-glibc2.35",
      "oqs_version": "0.14.1",
      "nacl_version": "1.6.2",
      "ascon_version": "0.2.0"
    }
  },
  "models": [
    {
      "name": "ModelA_Kyber512",
      "iterations": 100,
      "failures": 0,
      "encaps_ms": {
        "mean": 0.046,
        "stdev": 0.012
      },
      "decaps_ms": {
        "mean": 0.023,
        "stdev": 0.006
      },
      "kdf_ms": {
        "mean": 0.019,
        "stdev": 0.003
      },
      "enc_ms": {
        "mean": 0.758,
        "stdev": 0.045
      },
      "dec_ms": {
        "mean": 0.768,
        "stdev": 0.048
      },
      "total_ms": {
        "mean": 1.614,
        "stdev": 0.067
      }
    }
  ]
}
```

#### Field Meanings

| Field | Type | Unit | Meaning |
|-------|------|------|---------|
| `meta.iterations` | int | count | Number of repetitions per model |
| `meta.payload_bytes` | int | bytes | Size of plaintext |
| `meta.aad_bytes` | int | bytes | Additional authenticated data |
| `meta.seed` | int | — | Random seed for reproducibility |
| `models[0].name` | str | — | Model identifier (ModelA_Kyber512, etc.) |
| `models[0].iterations` | int | count | Successful iterations (100 - failures) |
| `models[0].failures` | int | count | Decryption mismatches |
| `encaps_ms.mean` | float | milliseconds | Average encapsulation time |
| `encaps_ms.stdev` | float | milliseconds | Standard deviation |
| `total_ms.mean` | float | milliseconds | Total latency (sum of all phases) |

#### Typical Ranges (from 100-iteration benchmark)

| Metric | ModelA | ModelB | ModelC | ModelD |
|--------|--------|--------|--------|--------|
| total_ms.mean | 1.614 | 2.266 | 1.701 | 1.914 |
| total_ms.stdev | 0.067 | 0.089 | 0.071 | 0.095 |
| encaps_ms.mean | 0.046 | 0.089 | 0.033 | 0.000 |
| decaps_ms.mean | 0.023 | 0.234 | 0.027 | 0.000 |

### CSV Output (`latency_results.csv`)

#### Structure

```csv
Model,Iteration,KeyGen_ns,Encaps_ns,Decaps_ns,KDF_ns,Encryption_ns,Decryption_ns,Total_ns,Failed
ModelA_Kyber512,1,0,90600,39900,71300,800800,739500,1742100,0
ModelA_Kyber512,2,0,63500,29600,20900,778799,810300,1703099,0
ModelA_Kyber512,3,0,71700,33300,19300,876300,747000,1747600,0
...
ModelB_BIKE_L1,1,0,76300,675900,16700,752200,800300,2321400,0
ModelB_BIKE_L1,2,0,109600,740900,29900,746800,728800,2356000,0
...
```

#### Column Definitions

| Column | Type | Unit | Meaning |
|--------|------|------|---------|
| `Model` | string | — | Model name |
| `Iteration` | int | count | Iteration number (1 to N) |
| `KeyGen_ns` | int | nanoseconds | Key generation (unused, always 0) |
| `Encaps_ns` | int | nanoseconds | KEM encapsulation time |
| `Decaps_ns` | int | nanoseconds | KEM decapsulation time |
| `KDF_ns` | int | nanoseconds | HKDF-SHA256 derivation time |
| `Encryption_ns` | int | nanoseconds | Ascon AEAD encryption |
| `Decryption_ns` | int | nanoseconds | Ascon AEAD decryption |
| `Total_ns` | int | nanoseconds | Sum of all phases |
| `Failed` | int | 0 or 1 | Decryption success (0=OK, 1=mismatch) |

#### CSV Validation Rules

```python
# Each row must satisfy:
Total_ns == Encaps_ns + Decaps_ns + KDF_ns + Encryption_ns + Decryption_ns

# Example validation:
import csv
with open('latency_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total = int(row['Total_ns'])
        computed = (int(row['Encaps_ns']) + int(row['Decaps_ns']) + 
                   int(row['KDF_ns']) + int(row['Encryption_ns']) + 
                   int(row['Decryption_ns']))
        assert total == computed, f"Row {row['Iteration']}: mismatch"
```

### Metrics Interpretation

#### Mean (Average)

**What it is:**
```
mean = (value1 + value2 + ... + valueN) / N
```

**Interpretation:**
- Center of the distribution
- Typical/expected latency
- What to expect on next run

**Example:**
```
encaps_ms.mean = 0.046 ms → On average, encapsulation takes 46 microseconds
```

#### Standard Deviation (Stdev)

**What it is:**
```
stdev = sqrt(sum((value - mean)^2) / (N-1))
```

**Interpretation:**
- Measures variability/jitter
- Low stdev (5-10% of mean) = consistent, predictable
- High stdev (>20% of mean) = variable, inconsistent

**Example:**
```
encaps_ms.mean = 0.046, stdev = 0.012
Interpretation: 68% of encapsulations take 0.034-0.058 ms (mean ± 1 stdev)
```

#### Coefficient of Variation (CV)

**Calculate:**
```
CV = (stdev / mean) * 100%
```

**Interpretation:**
| CV Range | Consistency | Action |
|----------|-------------|--------|
| <5% | Excellent | Use for timing predictions |
| 5-10% | Good | Acceptable |
| 10-20% | Fair | Increase iterations for better estimates |
| >20% | Poor | System noise; add warm-up discard |

#### Failures

**What it is:**
- Count of iterations where `decrypt(ciphertext) != plaintext`

**Interpretation:**
- Should always be 0
- If >0, indicates cryptographic bug or corruption

**Example:**
```
failures = 0  ✓ Correct
failures > 0  ✗ Error - investigate
```

---

## Iterations Detail

### Default Value

```bash
# In both scripts, default is 100 iterations:
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of iterations per model')
```

### Setting Iterations

```bash
# Fast test (1-2 seconds)
python main_experiment.py --iterations=10

# Standard run (5-10 seconds)
python main_experiment.py --iterations=100

# High precision (30-60 seconds)
python main_experiment.py --iterations=500

# Stress test (2+ minutes)
python main_experiment.py --iterations=1000
```

### Significance & Statistical Guidance

#### Low Iterations (1-50)

**Use Case:** Testing, quick validation

**Pros:**
- Fast execution
- Quick feedback

**Cons:**
- High variance (CPU warm-up, cache misses)
- Standard deviation large
- CV often >15%
- Not representative of steady-state

**Example (10 iterations):**
```
mean = 1.614 ms
stdev = 0.234 ms  ← Large uncertainty
CV = 14.5%
```

#### Standard Iterations (100-300)

**Use Case:** Typical benchmarking, publication

**Pros:**
- Good balance of speed and precision
- Steady-state reached
- CV typically 5-10%
- 1-2 minutes execution

**Cons:**
- Some warm-up effects still present
- Not suitable for sub-microsecond measurements

**Example (100 iterations):**
```
mean = 1.614 ms
stdev = 0.067 ms  ← Lower uncertainty
CV = 4.2%  ← Good
```

#### High Iterations (500-1000)

**Use Case:** High-precision benchmarks, statistical papers

**Pros:**
- Excellent precision
- CV < 3%
- Outliers averaged out
- Confidence intervals tight

**Cons:**
- 30-120 seconds execution
- May see different behavior over time (system load changes)

**Example (1000 iterations):**
```
mean = 1.614 ms
stdev = 0.031 ms  ← Very low uncertainty
CV = 1.9%  ← Excellent
Execution time: 1.614 seconds aggregate + overhead
```

### Choosing Iterations

**Decision Tree:**

```
Are you testing code?
  YES → 10-20 iterations (fast feedback)
  NO  → Continue...

Is this for a publication?
  YES → 500+ iterations (high precision)
  NO  → Continue...

Is performance critical (latency-sensitive app)?
  YES → 300+ iterations (steady-state)
  NO  → 100 iterations (standard)
```

### Warm-up Considerations

By default, **iterations are NOT discarded**. The first iteration may be slower due to:

- CPU cache cold state
- JIT compilation (if applicable)
- Memory allocation overhead

**If CV > 15% with 100 iterations:**

Option 1: Increase iterations
```bash
python main_experiment.py --iterations=500
```

Option 2: Manually discard first N (see Analysis section)

---

## Analysis Section

### Warm-up Discard Methods

#### Method 1: Command-Line Wrapper (Recommended)

```bash
# Run with extra iterations, then discard first 10
python main_experiment_enhanced.py --iterations=110
# Then analyze latency_results.csv, excluding first 10 rows per model
```

**Python code to discard:**
```python
import csv
import pandas as pd

df = pd.read_csv('latency_results.csv')

# Discard first 10 iterations per model
df_warmed = df.groupby('Model').apply(lambda x: x.iloc[10:]).reset_index(drop=True)

# Recompute stats
df_warmed.to_csv('latency_results_warmed.csv', index=False)
print(f"Kept {len(df_warmed)} rows, discarded {len(df) - len(df_warmed)}")
```

#### Method 2: Post-Processing with Pandas

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('latency_results.csv')

# Discard first 10 iterations per model
df_warmed = df[df['Iteration'] > 10]

# Recompute mean/stdev per model
for model in df_warmed['Model'].unique():
    model_data = df_warmed[df_warmed['Model'] == model]
    total_ns = model_data['Total_ns'].astype(int)
    
    mean_ms = total_ns.mean() / 1e6
    stdev_ms = total_ns.std() / 1e6
    
    print(f"{model}: {mean_ms:.3f} ± {stdev_ms:.3f} ms")
```

**Output:**
```
ModelA_Kyber512: 1.615 ± 0.063 ms
ModelB_BIKE_L1: 2.267 ± 0.081 ms
ModelC_X25519: 1.702 ± 0.068 ms
ModelD_PSK_Ascon80pq: 1.915 ± 0.087 ms
```

#### Method 3: Identify Inflection Point

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('latency_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
models = df['Model'].unique()

for idx, model in enumerate(models):
    ax = axes[idx // 2, idx % 2]
    model_data = df[df['Model'] == model].copy()
    
    # Plot rolling mean (window=10)
    model_data['rolling_mean'] = model_data['Total_ns'].rolling(window=10).mean()
    
    ax.plot(model_data['Iteration'], model_data['Total_ns'], alpha=0.3, label='Raw')
    ax.plot(model_data['Iteration'], model_data['rolling_mean'], 'r-', label='Rolling Mean (10)')
    ax.set_title(model)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total_ns')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig('warmup_analysis.png', dpi=150)
plt.show()

# Visual inspection shows where curve stabilizes
```

**Interpretation:**
- Flat curve = good, discard first ~5-10 rows
- Declining curve = increase iterations to reach plateau

### Plotting Instructions and Templates

#### Template 1: Single Model Latency Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('latency_results.csv')
model = 'ModelA_Kyber512'

data_ns = df[df['Model'] == model]['Total_ns'].astype(int)
data_ms = data_ns / 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(data_ms, bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Latency (ms)')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'{model} - Distribution')
axes[0].axvline(data_ms.mean(), color='r', linestyle='--', label=f'Mean: {data_ms.mean():.3f} ms')
axes[0].legend()
axes[0].grid(alpha=0.3)

# CDF (Cumulative Distribution Function)
sorted_data = np.sort(data_ms)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axes[1].plot(sorted_data, cdf, marker='.', linestyle='none')
axes[1].set_xlabel('Latency (ms)')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title(f'{model} - CDF')
axes[1].grid(alpha=0.3)

# Add percentile markers
for p in [50, 95, 99]:
    val = np.percentile(data_ms, p)
    axes[1].axvline(val, color='gray', linestyle=':', alpha=0.5)
    axes[1].text(val, 0.5, f'P{p}', rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig('latency_distribution.png', dpi=150)
plt.show()
```

#### Template 2: Multi-Model Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('latency_results.csv')

# Convert to milliseconds
df['Total_ms'] = df['Total_ns'] / 1e6

# Compute stats
stats = df.groupby('Model')['Total_ms'].agg(['mean', 'std', 'min', 'max']).reset_index()
stats = stats.sort_values('mean')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart with error bars
ax = axes[0]
ax.bar(range(len(stats)), stats['mean'], 
       yerr=stats['std'], capsize=5, alpha=0.7, 
       color=['C0', 'C1', 'C2', 'C3'])
ax.set_xticks(range(len(stats)))
ax.set_xticklabels(stats['Model'], rotation=45, ha='right')
ax.set_ylabel('Latency (ms)')
ax.set_title('Mean Latency with Std Dev')
ax.grid(alpha=0.3, axis='y')

# Add value labels
for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
    ax.text(i, mean + std + 0.05, f'{mean:.2f}±{std:.2f}', 
            ha='center', va='bottom', fontsize=9)

# Box plot
ax = axes[1]
data_by_model = [df[df['Model'] == m]['Total_ms'].values for m in df['Model'].unique()]
bp = ax.boxplot(data_by_model, labels=df['Model'].unique(), patch_artist=True)

# Color the boxes
for patch, color in zip(bp['boxes'], ['C0', 'C1', 'C2', 'C3']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Latency (ms)')
ax.set_title('Distribution (Box Plot)')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# Print table
print("\n" + "="*60)
print("LATENCY SUMMARY (ms)")
print("="*60)
print(stats.to_string(index=False))
print("="*60)
```

#### Template 3: Phase Breakdown (Enhanced CSV Only)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('latency_results.csv')

# Convert nanoseconds to microseconds for readability
phases = ['Encaps_ns', 'Decaps_ns', 'KDF_ns', 'Encryption_ns', 'Decryption_ns']
for phase in phases:
    df[phase.replace('_ns', '_us')] = df[phase] / 1000

phase_us = [p.replace('_ns', '_us') for p in phases]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model in enumerate(df['Model'].unique()):
    ax = axes[idx]
    model_data = df[df['Model'] == model][phase_us].mean()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    ax.barh(phase_us, model_data, color=colors)
    ax.set_xlabel('Average Time (µs)')
    ax.set_title(model)
    ax.grid(alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(model_data):
        ax.text(v + 5, i, f'{v:.1f}', va='center', fontsize=9)

plt.suptitle('Phase Breakdown by Model', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('phase_breakdown.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Template 4: Time Series (Iteration Trends)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('latency_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for idx, model in enumerate(df['Model'].unique()):
    ax = axes[idx]
    model_data = df[df['Model'] == model].copy()
    model_data['Total_ms'] = model_data['Total_ns'] / 1e6
    
    ax.plot(model_data['Iteration'], model_data['Total_ms'], 'b-', alpha=0.6, linewidth=1)
    ax.scatter(model_data['Iteration'], model_data['Total_ms'], s=10, alpha=0.3)
    
    mean = model_data['Total_ms'].mean()
    ax.axhline(mean, color='r', linestyle='--', label=f'Mean: {mean:.3f} ms')
    
    ax.fill_between(model_data['Iteration'], 
                    mean - model_data['Total_ms'].std(),
                    mean + model_data['Total_ms'].std(),
                    alpha=0.2, color='r', label='±1σ')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(model)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Latency Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('latency_timeseries.png', dpi=150)
plt.show()
```

#### Running the Templates

```bash
# Save any template as analysis.py
cat > analysis.py << 'EOF'
# [Paste template code here]
EOF

# Run it
python analysis.py
```

---

## .gitignore Template

Create a `.gitignore` file at project root to exclude build artifacts, outputs, and virtual environment:

```bash
cat > .gitignore << 'EOF'
# Virtual Environments
pqc_lab/
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
dist/
build/

# Benchmark Outputs
results.json
results_enhanced.json
latency_results.csv
latency_results_warmed.csv
*.png
*.pdf

# Analysis Artifacts
warmup_analysis.png
latency_distribution.png
model_comparison.png
phase_breakdown.png
latency_timeseries.png
analysis.py

# IDE/Editor
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# OS
.DS_Store
Thumbs.db
.env
.env.local

# Data Files (if large)
*.npy
*.npz

# Logs
*.log
*.out

# Temporary
tmp/
temp/
*.tmp
EOF
```

**File Locations:**

| File/Pattern | Why Exclude |
|--------------|------------|
| `pqc_lab/`, `venv/` | Virtual env is environment-specific, regenerated from requirements.txt |
| `__pycache__/`, `*.pyc` | Python bytecode, auto-generated |
| `results*.json`, `latency*.csv` | Benchmark outputs (large, generated) |
| `*.png` | Generated plots |
| `.vscode/`, `.idea/` | IDE config (personal preferences) |

**Verify:**
```bash
git status
# Should NOT show pqc_lab/, results.json, latency_results.csv, etc.
```

---

## Quick Reference

### Common Commands

#### Setup
```bash
# One-time setup
cd /path/to/Assignment_3
python3 -m venv pqc_lab
source pqc_lab/bin/activate  # Linux/WSL2
pip install --upgrade pip setuptools wheel
pip install oqs PyNaCl ascon numpy scipy pandas matplotlib
```

#### Run Benchmarks
```bash
# Quick test (10 iterations, ~2 seconds)
python main_experiment.py --iterations=10

# Standard (100 iterations, ~5 seconds)
python main_experiment.py --iterations=100

# Enhanced version with CSV
python main_experiment_enhanced.py --iterations=100 --csv-out=results_custom.csv

# Custom payload
python main_experiment.py --iterations=100 --payload-bytes=8192

# With AAD
python main_experiment.py --iterations=100 --aad="secret-context"

# Reproducible (same seed)
python main_experiment.py --iterations=100 --seed=42
```

#### Analysis
```bash
# Display results
cat results.json | python -m json.tool

# Quick CSV stats (with pandas)
python << 'EOF'
import pandas as pd
df = pd.read_csv('latency_results.csv')
print(df.groupby('Model')['Total_ns'].agg(['mean', 'std']))
EOF

# Run plotting template
python analysis.py
```

#### Troubleshooting
```bash
# Verify installation
python -c "
import oqs
from nacl.public import PrivateKey
from ascon import Ascon
print('✓ All dependencies OK')
"

# Check Python version
python --version

# Check venv activation
which python  # Should show pqc_lab/bin/python
```

### File Structure (After Running)

```
Assignment_3/
├── main_experiment.py                    # JSON-only version
├── main_experiment_enhanced.py           # JSON + CSV version
├── pqc_lab/                              # Virtual environment (in .gitignore)
├── results.json                          # Output: aggregated stats
├── latency_results.csv                   # Output: per-iteration data
├── MASTER_GUIDE.md                       # This file
├── EXPERIMENTAL_PIPELINE_SUMMARY.md      # Detailed pipeline docs
├── Structure.md                          # Project structure
├── .gitignore                            # Git exclusions
└── analysis.py                           # Your analysis script (custom)
```

### Key Metrics at a Glance

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| CV (Coefficient of Variation) | <5% | Excellent consistency |
| CV | 5-10% | Good consistency |
| CV | 10-20% | Fair, consider more iterations |
| CV | >20% | Poor, increase iterations or discard warm-up |
| Failures | 0 | Normal, crypto working correctly |
| Failures | >0 | Error, investigate implementation |

---

## Appendix: Platform-Specific Notes

### WSL2 Specific

**Performance:**
- WSL2 is nearly as fast as native Linux
- Use WSL2 for all development/testing
- Avoid native Windows (slower, harder to build)

**File Access:**
```bash
# Faster: Use Linux filesystem
cd ~/projects/Assignment_3

# Slower: Use mounted Windows filesystem
cd /mnt/d/Temp/SecurityInComputingandIT/Assignment_3
```

**Memory/Resource Limits** (in `.wslconfig`):
```ini
[interop]
appendWindowsPath = true

[wsl2]
memory=8GB
processors=4
swap=4GB
```

### macOS Specific

**Homebrew Install:**
```bash
brew install python3 openssl liboqs

# Set library path
export LDFLAGS="-L$(brew --prefix openssl)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include"
```

**M1/M2 Macs:**
- Use `python3.10+` (ARM64 native)
- liboqs wheels may not be available; build from source
- Generally slower for benchmarking due to emulation/compatibility mode

### Ubuntu 20.04 vs 22.04 vs 24.04

| Version | Python | liboqs | Notes |
|---------|--------|--------|-------|
| 20.04 LTS | 3.8 | 0.13 | Requires build from source for newer |
| 22.04 LTS | 3.10 | 0.14 | Good balance |
| 24.04 LTS | 3.12 | 0.14+ | Latest, fully supported |

**Recommendation:** Ubuntu 22.04 or WSL2 for best compatibility.

---

## Support & Further Reading

### Documentation Links
- **liboqs:** https://liboqs.org/
- **liboqs-python:** https://github.com/open-quantum-safe/liboqs-python
- **PyNaCl:** https://pynacl.readthedocs.io/
- **Ascon:** https://ascon.iaik.tugraz.at/
- **NIST PQC:** https://csrc.nist.gov/projects/post-quantum-cryptography

### Reporting Issues

If you encounter problems:

1. **Check this guide** (Dependency Troubleshooting section)
2. **Verify installations:**
   ```bash
   python -c "import oqs; print(oqs.Kem.get_supported()[:5])"
   ```
3. **Check versions:**
   ```bash
   pip list | grep -E "oqs|nacl|ascon"
   ```
4. **Check system libraries (Linux):**
   ```bash
   ldconfig -p | grep liboqs
   ```

### Performance Tuning

For publication-quality results:

1. **Warm-up:** Discard first 10-20 iterations
2. **Iterations:** Use 500+ for CV < 3%
3. **System idle:** Run on quiet system (close unnecessary apps)
4. **Affinity:** Pin to single core (advanced, Linux-only)
   ```bash
   taskset -c 0 python main_experiment.py --iterations=500
   ```

---

**End of MASTER_GUIDE.md**

Last Updated: January 2026
Version: 1.0
