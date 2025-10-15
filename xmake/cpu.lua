target("device-cpu")
    set_kind("static")
    add_files("../src/backend/device/cpu/*.cpp")
    on_install(function (target) end)
target_end()

target("ops-cpu")
    set_kind("static")
    add_deps("tensor")
    -- optimization and CPU-specific flags
    add_cxflags("-march=native", "-fopenmp", {force = true})

    -- link flags: keep -fopenmp for the linker as well
    add_ldflags("-fopenmp", {force = true})

    add_files("../src/backend/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()