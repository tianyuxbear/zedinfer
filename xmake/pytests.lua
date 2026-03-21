target("zedinfer_ops")
    set_kind("shared")
    set_prefixname("") -- drop 'lib' prefix for python module

    -- Deps & source
    add_deps("ops", "tensor", "core", "device", "utils")
    add_files("../tests/python/bindings/zedinfer_ops.cpp")
    add_packages("pybind11")

    -- oneDNN: link and copy libdnnl.so next to the output .so
    if has_config("onednn") then
        add_packages("onednn")

        -- $ORIGIN tells the dynamic linker: "look in the same directory as this .so"
        add_rpathdirs("$ORIGIN")

        -- After build: copy libdnnl.so.3 next to zedinfer_ops.so
        after_build(function (target)
            local onednn = target:pkg("onednn")
            if not onednn then return end
            local linkdirs = onednn:get("linkdirs")
            if not linkdirs then return end

            local outdir = target:targetdir()
            for _, dir in ipairs(linkdirs) do
                -- Copy all libdnnl*.so* files to the output directory
                local files = os.files(path.join(dir, "libdnnl*"))
                for _, f in ipairs(files) do
                    local dst = path.join(outdir, path.filename(f))
                    if not os.isfile(dst) then
                        os.cp(f, dst)
                        print("  -> copied %s to %s", path.filename(f), outdir)
                    end
                end
            end
        end)
    end

    -- Enable OpenMP shared linking
    add_shflags("-fopenmp")

    -- Output & install behavior
    set_targetdir("$(projectdir)/python/zedinfer") -- output to python pkg
    on_install(function (target) end)              -- skip default install
target_end()
