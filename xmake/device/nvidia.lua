target("device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    set_policy("check.auto_ignore_flags", false)
    add_cxflags("-fPIC", {force = true})
    add_cuflags("-rdc=true", "-Xcompiler=-fPIC", {force = true})
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    add_files("../../src/backend/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("ops-nvidia")
    set_kind("static")
    add_deps("tensor")
    set_policy("build.cuda.devlink", true)
    set_policy("check.auto_ignore_flags", false)
    add_cugencodes("sm_80", "sm_86", "sm_89", "sm_90", "sm_100")
    add_cxflags("-fPIC", {force = true})
    add_cuflags("-rdc=true", "-Xcompiler=-fPIC", {force = true})
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    add_files("../../src/backend/ops/*/nvidia/*.cu")

    -- cuBLAS for optimized linear (GEMM/GEMV)
    add_links("cublas", "cublasLt")

    -- cuDNN FlashAttention for prefill (optional)
    -- Requires cuDNN installed in CUDA_HOME (or /usr/local/cuda)
    -- Install: download cuDNN from https://developer.nvidia.com/cudnn-downloads
    --          and copy headers/libs to $CUDA_HOME/include and $CUDA_HOME/lib64
    if has_config("cudnn-flash") then
        add_links("cudnn", "cuda", "nvrtc")
    end

    on_install(function (target) end)
target_end()