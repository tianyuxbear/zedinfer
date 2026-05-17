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

    if has_config("flashinfer") then
        -- FlashInfer Mamba headers (selective_state_update.cuh and friends)
        -- need the BF16 conversion path enabled, host constexpr usable from
        -- __device__ code, and C++20-style templated lambdas in dispatch
        -- helpers. These mirror the flags proven by the M0 link probe target
        -- `test-flashinfer-ssu-link` (see xmake/tests.lua).
        add_defines("FLASHINFER_ENABLE_BF16")
        add_cuflags("--expt-relaxed-constexpr", "--expt-extended-lambda", {force = true})
    end

    on_install(function (target) end)
target_end()