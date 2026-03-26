-- Project configuration
set_project("zedinfer")
set_languages("c++17")
set_encodings("utf-8")

-- Build modes
add_rules("mode.debug", "mode.release")

if is_mode("debug") then
    add_defines("DEBUG")
end

-- Generate compile_commands.json for Clangd support
add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

-- Include directories
add_includedirs("include")
add_includedirs("third_party/include")

-- Platform-specific flags
if not is_plat("windows") then
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
end

if is_plat("windows") then
    add_cxxflags("/utf-8")
end

-- Enable compiler warnings
add_cxxflags("-Wall", "-Wextra")

-- Deps: Unicode processing
add_requires("icu4c")

-- Deps: Google Test framework (enable default main function)
add_requires("gtest", { configs = { main = true } })

-- Device implementations
includes("xmake/device/cpu.lua")

-- NVIDIA GPU support (optional)
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/device/nvidia.lua")
end

-- Portable build: use x86-64-v3 (AVX2) baseline instead of -march=native
-- for Docker/distribution builds that must run on different CPU generations.
-- oneDNN is unaffected (runtime ISA dispatch via JIT).
option("portable")
    set_default(false)
    set_showmenu(true)
    set_description("Build portable binaries with x86-64-v3 (AVX2) baseline instead of -march=native")
option_end()

-- oneDNN for optimized CPU linear (optional)
option("onednn")
    set_default(false)
    set_showmenu(true)
    set_description("Use oneDNN for optimized CPU linear (GEMM/GEMV)")
option_end()

if has_config("onednn") then
    add_requires("onednn")
    add_defines("USE_ONEDNN")
end


-- Python operator test bindings (optional)
option("pytest")
    set_default(false)
    set_showmenu(true)
    set_description("Build Python operator test bindings (requires pybind11)")
option_end()

if has_config("pytest") then
    add_requires("pybind11")
    includes("xmake/pytests.lua")
end

-- Test and example modules
includes("xmake/tests.lua")
includes("xmake/examples.lua")

-- Frontend and backend modules
includes("xmake/frontend.lua")
includes("xmake/backend.lua")

-- Utility library with CPU optimizations
target("utils")
    set_kind("static")
    if has_config("portable") then
        add_cxflags("-march=x86-64-v3", "-fopenmp", {force = true})
    else
        add_cxflags("-march=native", "-fopenmp", {force = true})
    end
    add_ldflags("-fopenmp", {force = true})
    add_files("src/utils/*.cpp")
    on_install(function (target) end)

-- Frontend library
target("frontend")
    set_kind("static")
    add_deps("tokenizer")
    add_deps("sampler")
    add_deps("loader")
    add_deps("models")
    on_install(function (target) end)
target_end()

-- Backend library
target("backend")
    set_kind("static")
    add_deps("core")
    add_deps("device")
    add_deps("tensor")
    add_deps("ops")
    add_deps("kvcache")
    on_install(function (target) end)
target_end()

-- Main library
target("zedinfer")
    set_kind("static")
    add_deps("utils")
    add_deps("frontend")
    add_deps("backend")
    add_files("src/zedinfer/*.cpp")
    on_install(function (target) end)
target_end()