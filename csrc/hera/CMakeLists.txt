find_package(Boost REQUIRED)

add_library(hera2py SHARED hera2py.cpp)

target_include_directories(hera2py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/upstream/include
    ${Boost_INCLUDE_DIRS}
)

target_compile_features(hera2py PRIVATE cxx_std_14)

target_compile_options(hera2py PRIVATE -O3 -march=native -flto)
target_link_options(hera2py PRIVATE -flto)

set_target_properties(hera2py PROPERTIES
    OUTPUT_NAME "hera2py"
    PREFIX "lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
