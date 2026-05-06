add_library(unionfind2py SHARED unionfind2py.cpp)

target_link_libraries(unionfind2py PRIVATE project_options)

set_target_properties(unionfind2py PROPERTIES
    OUTPUT_NAME "unionfind2py"
    PREFIX "lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
