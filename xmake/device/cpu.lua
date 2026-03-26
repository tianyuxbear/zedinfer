target("device-cpu")
    set_kind("static")
    add_files("../../src/backend/device/cpu/*.cpp")
    on_install(function (target) end)
target_end()

target("ops-cpu")
    set_kind("static")
    add_deps("tensor")
    -- CPU-specific and OpenMP flags
    if has_config("portable") then
        add_cxflags("-march=x86-64-v3", "-fopenmp", {force = true})
    else
        add_cxflags("-march=native", "-fopenmp", {force = true})
    end

    -- link flags: keep -fopenmp for the linker as well
    add_ldflags("-fopenmp", {force = true})

    add_files("../../src/backend/ops/*/cpu/*.cpp")
    -- rearrange/cpu/*.cpp is already compiled in the tensor target
    remove_files("../../src/backend/ops/rearrange/cpu/*.cpp")

    -- oneDNN support (optional)
    if has_config("onednn") then
        add_packages("onednn")
    end

    on_install(function (target) end)
target_end()
