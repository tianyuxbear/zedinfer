# CUDA Profiling and Debugging Guide

> Practical reference for analyzing and debugging CUDA programs in zedinfer.
> Based on real debugging sessions with nsys, compute-sanitizer, and other tools.

---

## 1. Nsight Systems (nsys) — Performance Profiling

### What it does
Traces GPU kernel launches, CUDA API calls, CPU-GPU synchronization, and memory transfers. Shows where time is spent.

### Install
Part of CUDA Toolkit. Located at `$CUDA_HOME/bin/nsys`.

### Basic profiling

```bash
# Profile a single run, save to file
nsys profile --stats=true -o /tmp/my_profile \
    xmake run bench /path/to/model --nvidia -p 1 -d 128 -r 1 --gpu-memory-utilization 0.1
```

### Analyzing results

```bash
# GPU kernel time summary (most useful — shows which kernels dominate)
nsys stats --report cuda_gpu_kern_sum /tmp/my_profile.nsys-rep

# CUDA API call summary (shows CPU-side overhead: launch, memcpy, malloc)
nsys stats --report cuda_api_sum /tmp/my_profile.nsys-rep

# Memory operations summary
nsys stats --report cuda_gpu_mem_size_sum /tmp/my_profile.nsys-rep

# Full timeline (open in Nsight Systems GUI)
# Copy .nsys-rep file to local machine, open with nsys-ui
```

### What to look for

**Kernel summary (`cuda_gpu_kern_sum`):**
- Which kernel takes the most total time? (attention? cuBLAS GEMV?)
- Average per-call time — is it reasonable for the problem size?
- High stddev — indicates inconsistent performance (cold start? memory pressure?)
- Max time outliers — first calls may include JIT compilation

**API summary (`cuda_api_sum`):**
- `cudaMallocHost` / `cudaFreeHost` — expensive if called per step (should be pre-allocated)
- `cudaLaunchKernel` — average launch overhead (typically 2-5us, >10us is suspicious)
- `cudaMemcpy` — synchronous copies block CPU (should use async where possible)
- `cudaStreamSynchronize` — implicit sync points (D2H copies, argmax result readback)

### Real example from zedinfer

We discovered a 23% decode throughput difference between builds. nsys revealed:

```
Without cuDNN build:
  cudaMallocHost: 218.8ms total, 384 calls, avg 570us/call  ← ROOT CAUSE
  cudaFreeHost:    94.4ms total, 384 calls, avg 246us/call

With cuDNN build:
  cudaMallocHost:   2.3ms total, 384 calls, avg 6us/call
  cudaFreeHost:     1.8ms total, 384 calls, avg 5us/call
```

The fix: pre-allocate pinned host buffer in ArgmaxSampler instead of per-call cudaMallocHost.

---

## 2. Compute Sanitizer — Memory Error Detection

### What it does
Detects illegal memory accesses, race conditions, and uninitialized memory reads in CUDA kernels. Similar to Valgrind for CPU code.

### Basic memory check

```bash
# Build in debug mode for better error messages with source line info
xmake f -m debug --nv-gpu=y --onednn=y
xmake build ping

# Run with memory checker
compute-sanitizer --tool memcheck \
    xmake run ping /path/to/model --nvidia --gpu-memory-utilization 0.3
```

### Tools available

| Tool | Purpose | Usage |
|------|---------|-------|
| `memcheck` | Illegal memory access, out-of-bounds, misaligned | Default, most common |
| `racecheck` | Shared memory race conditions | `--tool racecheck` |
| `initcheck` | Uninitialized device memory reads | `--tool initcheck` |
| `synccheck` | Barrier/sync errors in cooperative groups | `--tool synccheck` |

### What the output tells you

```
========= Invalid __global__ read of size 4 bytes
=========     at kernel_name+0x2860 in file.cu:418        ← WHICH kernel, WHICH line
=========     by thread (0,0,0) in block (0,0,0)          ← WHICH thread
=========     Access to 0x56f503254cb0 is out of bounds    ← WHAT address
=========     and is 35TB bytes before the nearest allocation at 0x77038a000000
```

Key clues:
- **Address range `0x56xx` or `0x7fxx`** = HOST memory pointer passed to GPU kernel (HMM issue)
- **Address range `0x7x03xxxxxxxx`** = GPU memory, but out of bounds (buffer overflow)
- **"before the nearest allocation"** = pointer is completely wrong (uninitialized or host pointer)
- **"after the nearest allocation"** = buffer overflow (index out of bounds)

### Real example from zedinfer

RTX 4090 crashed during warmup. compute-sanitizer revealed:

```
Invalid __global__ read of size 4 bytes
    at paged_attention_prefill_kernel<bf16>+0x2860 in paged_attention_nvidia.cu:418
    Access to 0x56f503254cb0 is out of bounds
    and is 35TB bytes before the nearest allocation at 0x77038a000000
```

Address `0x56f5...` is a CPU address → the block table (host `std::vector::data()`) was passed directly to a GPU kernel. B200/A6000 had HMM support so it "worked"; RTX 4090 without HMM crashed.

Fix: upload block table to GPU tensor before passing to kernel.

---

## 3. Checking for Pending CUDA Errors

CUDA errors from async operations (kernel launches, async memcpy) don't surface immediately. They accumulate and cause later operations to fail with cryptic errors.

### Pattern: check before suspect operation

```cpp
cudaError_t priorErr = cudaGetLastError();
if (priorErr != cudaSuccess) {
    fprintf(stderr, "Pending CUDA error: %s\n", cudaGetErrorString(priorErr));
}
// Now run the suspect operation
```

### Pattern: sync and check after kernel launch

```cpp
myKernel<<<grid, block>>>(...);
cudaError_t err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
}
```

### Common error codes

| Error | Meaning | Typical cause |
|-------|---------|--------------|
| `illegal memory access` | Kernel read/wrote invalid address | Host pointer to GPU kernel, buffer overflow |
| `misaligned address` | Unaligned memory access | Wrong tensor stride, bad pointer arithmetic |
| `launch failed` | Kernel crashed | Stack overflow, infinite loop, illegal instruction |
| `out of memory` | cudaMalloc failed | Pool too large, memory leak |
| `invalid device function` | Kernel not compiled for this GPU | Missing gencode for target SM |

---

## 4. cuBLAS Error Diagnosis

### Status codes

| Code | Name | Meaning |
|------|------|---------|
| 1 | NOT_INITIALIZED | CUDA context not active when handle created |
| 3 | ALLOC_FAILED | Workspace allocation failed |
| 7 | INVALID_VALUE | Bad parameter (dimensions, nullptr) |
| 8 | ARCH_MISMATCH | Kernel not available for this GPU |
| 13 | EXECUTION_FAILED | Kernel crashed during execution |
| 14 | INTERNAL_ERROR | Catch-all — often caused by pending CUDA error from earlier |
| 15 | NOT_SUPPORTED | Algorithm not supported for this configuration |

### Debugging cuBLAS failures

```cpp
// Add shape info to error messages
if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "cuBLASLt failed: status=" + std::to_string(status) +
        " M=" + std::to_string(M) + " N=" + std::to_string(N) +
        " K=" + std::to_string(K) + " dtype=" + std::to_string(type));
}
```

### Key insight
**cuBLAS status 14 (INTERNAL_ERROR) is often NOT a cuBLAS bug** — it's caused by a pending CUDA error from a previous kernel. Always check `cudaGetLastError()` before blaming cuBLAS.

---

## 5. GPU Memory Debugging

### Check available memory

```cpp
size_t free, total;
cudaMemGetInfo(&free, &total);
printf("GPU memory: %zu MB free / %zu MB total\n", free/1024/1024, total/1024/1024);
```

### nvidia-smi monitoring

```bash
# One-shot
nvidia-smi

# Continuous monitoring (every 1 second)
watch -n 1 nvidia-smi

# Memory usage by process
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### Detect memory leaks

```bash
# Run with cuda-memcheck leak detection
compute-sanitizer --tool memcheck --leak-check full \
    xmake run ping /path/to/model --nvidia
```

---

## 6. Kernel Launch Configuration Debugging

### Check occupancy

```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
printf("Suggested: grid=%d, block=%d\n", minGridSize, blockSize);
```

### Check shared memory limits

```bash
# Show GPU properties including max shared memory per block
nvidia-smi -q | grep -i "shared"
```

### Common launch issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `invalid configuration argument` | Block size > 1024 or shared memory > limit | Reduce block size or shared memory |
| Kernel runs but wrong results | Grid/block dims don't cover all data | Check grid calculation: `(N + BLOCK - 1) / BLOCK` |
| Very slow kernel | Low occupancy | Reduce register usage or shared memory |

---

## 7. Environment Variables for Debugging

```bash
# Enable CUDA error checking (sync after every API call — very slow but catches errors early)
export CUDA_LAUNCH_BLOCKING=1

# Verbose cuBLAS logging
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Force specific GPU
export CUDA_VISIBLE_DEVICES=0

# Disable CUDA caching allocator (helps find memory bugs)
export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # (PyTorch only, not applicable to zedinfer)
```

---

## 8. Performance Optimization Workflow

### Step 1: Identify the bottleneck

```bash
nsys profile --stats=true -o /tmp/prof xmake run bench ... -r 1
nsys stats --report cuda_gpu_kern_sum /tmp/prof.nsys-rep
```

Look at the top kernel by total time percentage.

### Step 2: Understand the kernel

- Is it compute-bound or memory-bound?
- What is the theoretical throughput vs actual?
- What is the GPU utilization? (12 blocks on 128 SMs = 9% utilization)

### Step 3: Profile the kernel in detail

```bash
# Nsight Compute for detailed kernel analysis
ncu --set full -o /tmp/kernel_prof xmake run bench ... -r 1
```

Nsight Compute shows:
- Warp occupancy
- Memory throughput (L1, L2, HBM)
- Compute throughput (FP32, FP16, tensor cores)
- Stall reasons (memory latency, instruction fetch, etc.)

### Step 4: Optimize and measure

Always measure before and after with the same nsys config.

---

## 9. Quick Reference Commands

```bash
# Profile end-to-end
nsys profile --stats=true -o /tmp/prof xmake run bench /path/to/model --nvidia -p 1 -d 128 -r 1

# Kernel time breakdown
nsys stats --report cuda_gpu_kern_sum /tmp/prof.nsys-rep

# API overhead breakdown
nsys stats --report cuda_api_sum /tmp/prof.nsys-rep

# Memory error check
compute-sanitizer --tool memcheck xmake run ping /path/to/model --nvidia 2>&1 | head -50

# Race condition check (shared memory)
compute-sanitizer --tool racecheck xmake run ping /path/to/model --nvidia 2>&1 | head -50

# Check pending CUDA errors in code
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

# Monitor GPU usage
watch -n 1 nvidia-smi

# Force synchronous CUDA execution (debug only, very slow)
CUDA_LAUNCH_BLOCKING=1 xmake run ping /path/to/model --nvidia
```
