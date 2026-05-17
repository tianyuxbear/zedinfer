local termbar_root = path.absolute("third_party/termbar", os.projectdir())
local termbar_include = path.join(termbar_root, "include")
local termbar_src = path.join(termbar_root, "src", "termbar.cpp")
local has_termbar = os.isdir(termbar_include) and os.isfile(termbar_src)

if not has_termbar then
    os.raise("required termbar submodule not found: %s\nRun `git submodule update --init --recursive` to fetch third-party dependencies.", termbar_root)
end

target("termbar")
    set_kind("static")
    add_files(termbar_src)
    add_includedirs(termbar_include, {public = true})
    add_defines("ZEDINFER_USE_TERMBAR", {public = true})
    if not is_plat("windows") then
        add_syslinks("pthread")
    end
    on_install(function (target) end)
target_end()

target("tokenizer")
    set_kind("static")
    add_packages("icu4c", {public = true})
    -- Enable OpenMP support for multi-threaded BPE encoding.
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
    add_files("../src/frontend/tokenizer/*.cpp")
    on_install(function (target) end)
target_end()    

target("sampler")
    set_kind("static")
    add_files("../src/frontend/sampler/*.cpp")
    on_install(function (target) end)
target_end()

target("loader")
    set_kind("static")
    add_files("../src/frontend/loader/*.cpp")
    on_install(function (target) end)
target_end()

target("models")
    set_kind("static")
    add_deps("loader")
    add_deps("termbar")
    add_files("../src/frontend/models/*.cpp")
    -- ChatTemplateJinja is referenced by Qwen3_5Model::Qwen3_5Model in this
    -- target. Linking it from the parent `zedinfer` archive would require the
    -- linker to revisit `zedinfer.a` after `models.a` resolves its undefined
    -- symbol, which the default -lzedinfer ... -lmodels order forbids. Compile
    -- the TU here so `models.a` carries its own dependency.
    add_files("../src/zedinfer/chat_template_jinja.cpp")
    on_install(function (target) end)
target_end()
