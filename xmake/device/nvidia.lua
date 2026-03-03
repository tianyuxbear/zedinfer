target("device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    add_cxflags("-fPIC", {force = true})
    add_cuflags("-Xcompiler=-fPIC", {force = true})
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    add_files("../../src/backend/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("ops-nvidia")
    set_kind("static")
    add_deps("tensor")
    set_policy("build.cuda.devlink", true)
    add_cugencodes("sm_89")
    add_cxflags("-fPIC", {force = true})
    add_cuflags("-Xcompiler=-fPIC", {force = true})
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    add_files("../../src/backend/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()