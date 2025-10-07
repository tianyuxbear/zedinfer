add_requires("gtest")

target("test")
    set_kind("binary")
    add_deps("tokenizer")
    add_files("../test/tokenizer/*.cpp")
    add_packages("gtest")
    add_syslinks("gtest_main") 