target("zedinfer_ops")
    set_kind("shared")
    set_prefixname("") -- drop 'lib' prefix for python module

    -- Deps & source
    add_deps("ops", "tensor", "core", "device", "utils")
    add_files("../tests/python/bindings/zedinfer_ops.cpp")
    add_packages("pybind11")

    -- Enable OpenMP shared linking
    add_shflags("-fopenmp")  -- CRITICAL: Use shflags for shared libraries!
    
    -- Output & install behavior
    set_targetdir("$(projectdir)/python/zedinfer") -- output to python pkg
    on_install(function (target) end)              -- skip default install
target_end()