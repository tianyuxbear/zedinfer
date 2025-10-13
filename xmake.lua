-- 设置项目名称
set_project("neollm")

-- 设置C++标准
set_languages("c++17")

-- 设置源文件编码
set_encodings("utf-8")

-- 全局编译模式
add_rules("mode.debug", "mode.release")

-- 自动生成 compile_commands.json 文件到 build 目录
add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

-- 全局添加 include 目录到头文件搜索路径
add_includedirs("include")
add_includedirs("third_party/include")


if not is_plat("windows") then
    -- -fPIC: 生成位置无关代码，用于动态库
    -- -Wno-unknown-pragmas: 忽略未知的 #pragma 指令警告
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
end

-- Windows平台特殊设置
if is_plat("windows") then
    add_cxxflags("/utf-8")
end

-- 启用所有常规警告 + 额外警告
-- -Wall: 启用大部分警告
-- -Wextra: 启用额外的合理警告
add_cxxflags("-Wall", "-Wextra")

add_requires("icu4c")

target("utils")
    set_kind("static")
    add_files("src/utils/*.cpp")

target("tokenizer")
    set_kind("static")
    add_packages("icu4c")
    add_files("src/frontend/tokenizer/*.cpp")

target("loader")
    set_kind("static")
    add_deps("utils")
    add_files("src/frontend/loader/*.cpp")

target("core")
    set_kind("static")
    add_files("src/backend/core/**.cpp")

includes("xmake/test.lua")
