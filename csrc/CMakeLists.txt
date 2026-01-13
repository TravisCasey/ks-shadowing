find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(ks2py SHARED
    ksint2py.cpp
    ksint.cpp
)

target_include_directories(ks2py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(ks2py PRIVATE
    ${FFTW3_LIBRARIES}
    Eigen3::Eigen
)

target_compile_options(ks2py PRIVATE
    -O3
    -march=native
    -ffast-math
)

set_target_properties(ks2py PROPERTIES
    OUTPUT_NAME "ks2py"
    PREFIX "lib"
)
