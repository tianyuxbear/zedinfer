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

local function zedinfer_trim(s)
    return s and s:trim() or nil
end

local function zedinfer_first_line(s)
    if not s or s == "" then
        return nil
    end
    return s:match("([^\r\n]+)")
end

local function zedinfer_c_string(s)
    s = tostring(s or "unknown")
    s = s:gsub("\\", "\\\\")
    s = s:gsub("\"", "\\\"")
    return s
end

local function zedinfer_build_version(git)
    local git_hash = git({"rev-parse", "--short", "HEAD"}) or os.getenv("ZEDINFER_GIT_HASH") or "unknown"
    local tag = zedinfer_first_line(git({"tag", "--points-at", "HEAD", "--sort=-version:refname"}))
    local base = tag or git_hash
    if base == "unknown" and os.getenv("ZEDINFER_VERSION") then
        base = os.getenv("ZEDINFER_VERSION")
    end

    local dirty = false
    if git({"rev-parse", "--is-inside-work-tree"}) == "true" then
        local status = git({"status", "--porcelain"})
        dirty = status ~= nil and status ~= ""
    end

    if dirty then
        base = base .. "-dirty"
    end

    return base, git_hash, os.date("%Y-%m-%d")
end

-- Include directories
add_includedirs("include")

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

option("flashinfer")
    set_default(false)
    set_showmenu(true)
    set_description("Use local FlashInfer headers as the NVIDIA paged attention backend")
option_end()

local project_dir = os.projectdir()
local third_party_hint = "Run `git submodule update --init --recursive` to fetch third-party dependencies."

local function add_required_includedir(paths, hint)
    local candidates = type(paths) == "table" and paths or {paths}
    for _, relpath in ipairs(candidates) do
        local abspath = path.absolute(relpath, project_dir)
        if os.isdir(abspath) then
            add_includedirs(abspath)
            return
        end
    end
    os.raise("required include directory not found: %s\n%s", table.concat(candidates, ", "), hint)
end

add_required_includedir("third_party/nlohmann_json/single_include", third_party_hint)
add_required_includedir("third_party/plog/include", third_party_hint)
add_required_includedir("third_party/argparse/include", third_party_hint)
add_required_includedir("third_party/cpp-httplib", third_party_hint)
add_required_includedir("third_party/dbg-macro", third_party_hint)
add_required_includedir("third_party/minja/include", third_party_hint)
add_required_includedir("third_party/stb", third_party_hint)

if has_config("flashinfer") then
    add_defines("USE_FLASHINFER")
    local flashinfer_hint = "Run `git submodule update --init --recursive` to fetch FlashInfer and its nested dependencies."
    add_required_includedir("third_party/flashinfer/include", flashinfer_hint)
    add_required_includedir({
        "third_party/flashinfer/3rdparty/cutlass/include",
        "third_party/cutlass/include",
    }, flashinfer_hint)
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
    -- chat_template_jinja.cpp is compiled into the `models` target so that
    -- Qwen3_5Model can call ChatTemplateJinja::load without violating the
    -- static-link order (models is downstream of zedinfer in -l flags).
    add_files("src/zedinfer/*.cpp|chat_template_jinja.cpp")

    -- Inject build identity as compile-time defines.
    -- If HEAD has a tag and the worktree is clean, the version is that tag.
    -- Otherwise it is the short commit hash. Dirty worktrees append "-dirty".
    on_config(function (target)
        local function git(args)
            return zedinfer_trim(try { function() return os.iorunv("git", args) end })
        end
        local version, git_hash, build_date = zedinfer_build_version(git)
        target:add("defines", 'ZEDINFER_VERSION="' .. zedinfer_c_string(version) .. '"', {public = true})
        target:add("defines", 'ZEDINFER_GIT_HASH="' .. zedinfer_c_string(git_hash) .. '"', {public = true})
        target:add("defines", 'ZEDINFER_BUILD_DATE="' .. zedinfer_c_string(build_date) .. '"', {public = true})
    end)

    on_install(function (target) end)
target_end()
