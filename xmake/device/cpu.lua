target("device-cpu")
    set_kind("static")
    add_files("../../src/backend/device/cpu/*.cpp")
    on_install(function (target) end)
target_end()

target("ops-cpu")
    set_kind("static")
    add_deps("tensor")
    -- CPU-specific and OpenMP flags
    add_cxflags("-march=native", "-fopenmp", {force = true})

    -- link flags: keep -fopenmp for the linker as well
    add_ldflags("-fopenmp", {force = true})

    add_files("../../src/backend/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()