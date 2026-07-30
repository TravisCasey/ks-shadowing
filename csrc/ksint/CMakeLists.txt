find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(ks2py SHARED
    ksint2py.cpp
    ksint.cpp
)

target_include_directories(ks2py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ks2py PRIVATE
    project_options
    PkgConfig::FFTW3
    Eigen3::Eigen
)

target_compile_options(ks2py PRIVATE -ffast-math)

set_target_properties(ks2py PROPERTIES
    OUTPUT_NAME "ks2py"
    PREFIX "lib"
    # macOS defaults SHARED libraries to .dylib; the ctypes loaders and the
    # hatch build hook both address these by an explicit libNAME.so filename.
    SUFFIX ".so"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
