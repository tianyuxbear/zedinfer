target("device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)

    add_files("../../src/backend/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("ops-nvidia")
    set_kind("static")
    add_deps("tensor")
    set_policy("build.cuda.devlink", true)
    add_cugencodes("native")

    add_files("../../src/backend/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()