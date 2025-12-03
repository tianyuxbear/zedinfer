target("device")
    set_kind("static")
    add_deps("utils")
    add_deps("device-cpu")
    if has_config("nv-gpu") then
        add_deps("device-nvidia")
    end
    add_files("../src/backend/device/*.cpp")
    on_install(function (target) end)
target_end()

target("core")
    set_kind("static")
    add_deps("device")
    add_files("../src/backend/core/**.cpp")
    on_install(function (target) end)
target_end()

target("tensor")
    set_kind("static")
    add_deps("utils")
    add_deps("core")
    add_files("../src/backend/tensor/*.cpp")
    add_files("../src/backend/ops/rearrange/*.cpp")
    add_files("../src/backend/ops/rearrange/cpu/*.cpp")
    on_install(function (target) end)
target_end()

target("ops")
    set_kind("static")
    add_deps("ops-cpu")
    if has_config("nv-gpu") then
        add_deps("ops-nvidia")
    end
    add_files("../src/backend/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

target("kvcache")
    set_kind("static")
    add_deps("tensor")
    add_files("../src/backend/kvcache/*.cpp")
    on_install(function (target) end)
target_end()

