find_package(Boost REQUIRED CONFIG)

add_library(hera2py SHARED hera2py.cpp)

target_include_directories(hera2py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/upstream/include
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(hera2py PRIVATE project_options)

set_target_properties(hera2py PROPERTIES
    OUTPUT_NAME "hera2py"
    PREFIX "lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
