# Benchmark Results on B200 180GB VRAM

**Test machine:** B200, **180GB VRAM**

**Model:** `~/data/models/DeepSeek-R1-Distill-Qwen-1.5B`

---

## 1. Build Without cuDNN

### Build Steps

```bash
# Clean old compiled files
xmake clean --all
rm -rf build .xmake

# Configure and build without cuDNN
xmake f -m release --nv-gpu=y --onednn=y --pytest=y
```

```bash
xmake build -v bench > logs/build_bench_without_cudnn.log 2>&1
```

**Build log:** `logs/build_bench_without_cudnn.log`

---

### Benchmark 1

```bash
xmake run bench ~/data/models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 128 -d 128 -r 3 --gpu-memory-utilization 0.1
```

**Result:**

```plaintext
[ZedInfer] Running 3 rounds of profiling...

================ Performance Report ================
Config : prefill=128, decode=128, rounds=3

Prefill:     4.13 ms | 30980.77 tok/s
Decode :   541.06 ms |   236.57 tok/s
====================================================
```

---

### Benchmark 2

```bash
xmake run bench ~/data/models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 1 -d 128 -r 3 --gpu-memory-utilization 0.1
```

**Result:**

```plaintext
[ZedInfer] Running 3 rounds of profiling...

================ Performance Report ================
Config : prefill=1, decode=128, rounds=3

Prefill:     3.08 ms |   325.16 tok/s
Decode :   443.67 ms |   288.50 tok/s
====================================================
```

---

## 2. Build With cuDNN

### Build Steps

```bash
# Clean old compiled files
xmake clean --all
rm -rf build .xmake

# Configure and build with cuDNN Flash Attention enabled
xmake f -m release --nv-gpu=y --onednn=y --cudnn-flash=y --pytest=y
```

```bash
xmake build -v bench > logs/build_bench_with_cudnn.log 2>&1
```

**Build log:** `logs/build_bench_with_cudnn.log`

---

### Benchmark 1

```bash
xmake run bench ~/data/models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 128 -d 128 -r 3 --gpu-memory-utilization 0.1
```

**Result:**

```plaintext
[ZedInfer] Running 3 rounds of profiling...

================ Performance Report ================
Config : prefill=128, decode=128, rounds=3

Prefill:   116.24 ms |  1101.20 tok/s
Decode :   435.41 ms |   293.98 tok/s
====================================================
```

---

### Benchmark 2

```bash
xmake run bench ~/data/models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 1 -d 128 -r 3 --gpu-memory-utilization 0.1
```

**Result:**

```plaintext
[ZedInfer] Running 3 rounds of profiling...

================ Performance Report ================
Config : prefill=1, decode=128, rounds=3

Prefill:     2.33 ms |   429.93 tok/s
Decode :   343.24 ms |   372.91 tok/s
====================================================
```

---

## 3. Summary

### Without cuDNN
- For `prefill=128, decode=128`:
  - Prefill: **4.13 ms**, **30980.77 tok/s**
  - Decode: **541.06 ms**, **236.57 tok/s**

- For `prefill=1, decode=128`:
  - Prefill: **3.08 ms**, **325.16 tok/s**
  - Decode: **443.67 ms**, **288.50 tok/s**

### With cuDNN
- For `prefill=128, decode=128`:
  - Prefill: **116.24 ms**, **1101.20 tok/s**
  - Decode: **435.41 ms**, **293.98 tok/s**

- For `prefill=1, decode=128`:
  - Prefill: **2.33 ms**, **429.93 tok/s**
  - Decode: **343.24 ms**, **372.91 tok/s**

---

## 4. Observations

- Enabling cuDNN significantly improved **decode** performance in both test cases.
- For the `prefill=1` case, cuDNN also improved **prefill** latency.
- However, for the `prefill=128` case, **prefill performance dropped significantly** after enabling cuDNN:
  - Without cuDNN: **4.13 ms**
  - With cuDNN: **116.24 ms**

This suggests that cuDNN helps decode performance, but may introduce a major regression in large-prefill scenarios.