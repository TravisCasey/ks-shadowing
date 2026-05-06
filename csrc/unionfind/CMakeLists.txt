add_library(unionfind2py SHARED unionfind2py.cpp)

target_compile_options(unionfind2py PRIVATE
    -O3
    -march=x86-64-v3
    -Wall
    -Wextra
    -Wpedantic
)

set_target_properties(unionfind2py PROPERTIES
    OUTPUT_NAME "unionfind2py"
    PREFIX "lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
