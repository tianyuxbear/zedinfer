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
    -- MoE ops CUDA files (flat layout alongside op.cpp)
    add_files("../../src/backend/ops/moe/*.cu")

    -- cuBLAS for optimized linear (GEMM/GEMV)
    add_links("cublas", "cublasLt")


    on_install(function (target) end)
target_end()