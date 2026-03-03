-- Python operator testing module (pybind11)
-- Build: xmake f --pytest=y && xmake build zedinfer_ops
-- The output shared library can be imported directly in Python.

option("pytest")
    set_default(false)
    set_showmenu(true)
    set_description("Build Python operator test bindings (requires pybind11 and Python)")
option_end()

if has_config("pytest") then
    add_requires("pybind11")

    target("zedinfer_ops")
        set_kind("shared")
        add_rules("python.library", "pybind11")
        add_files("../tests/python/bindings/zedinfer_ops.cpp")
        add_deps("ops", "tensor", "core", "device", "utils")
        add_packages("pybind11")

        -- CPU optimization flags consistent with project
        add_cxflags("-march=native", "-fopenmp", {force = true})
        add_ldflags("-fopenmp", {force = true})

        -- Output to tests/python/ so Python can import it directly
        set_targetdir("$(projectdir)/tests/python")

        on_install(function (target) end)
    target_end()
end
