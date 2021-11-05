cmake_minimum_required(VERSION 3.15...3.19)

project(scikit_build_example VERSION "0.0.1")

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_REGISTRY "LAST")
set(Python3_FIND_FRAMEWORK "LAST")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pdal CONFIG REQUIRED)
find_package(Arrow REQUIRED)
#find_package(ArrowPython REQUIRED)
# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${PDAL_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE ${Python3_NumPy_LIBRARIES} ${PDAL_LIBRARIES} arrow_shared)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION .)
