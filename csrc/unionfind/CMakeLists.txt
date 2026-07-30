add_library(unionfind2py SHARED unionfind2py.cpp)

target_link_libraries(unionfind2py PRIVATE project_options)

set_target_properties(unionfind2py PROPERTIES
    OUTPUT_NAME "unionfind2py"
    PREFIX "lib"
    # macOS defaults SHARED libraries to .dylib; the ctypes loaders and the
    # hatch build hook both address these by an explicit libNAME.so filename.
    SUFFIX ".so"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
