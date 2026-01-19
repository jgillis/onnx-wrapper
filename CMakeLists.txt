# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)

project(onnx-wrapper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position independent code (required for shared library)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Install directories (needed early for generator expressions)
include(GNUInstallDirs)

# Dependencies
include(ExternalProject)

# Try to find system dependencies
# Dependency chain: ONNX -> Protobuf -> (absl, utf8_range)
find_package(absl QUIET)
find_package(utf8_range QUIET)
find_package(Protobuf QUIET)
find_package(ONNX QUIET)

# Check what we can use from the system
if(ONNX_FOUND AND Protobuf_FOUND AND absl_FOUND AND utf8_range_FOUND)
    # Case 1: Use everything from system
    message(STATUS "Found system ONNX: ${ONNX_VERSION}")
    message(STATUS "Found system Protobuf: ${Protobuf_VERSION}")
    set(USE_SYSTEM_ONNX TRUE)
    set(USE_SYSTEM_PROTOBUF TRUE)
elseif(Protobuf_FOUND AND absl_FOUND AND utf8_range_FOUND)
    # Case 2: Use system Protobuf, build ONNX from source
    message(STATUS "Found system Protobuf: ${Protobuf_VERSION}")
    message(STATUS "ONNX not found, building from source")
    set(USE_SYSTEM_ONNX FALSE)
    set(USE_SYSTEM_PROTOBUF TRUE)
    set(ONNX_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
else()
    # Case 3: Build everything from source
    message(STATUS "System ONNX and Protobuf not found, building from source")
    set(USE_SYSTEM_ONNX FALSE)
    set(USE_SYSTEM_PROTOBUF FALSE)
    set(ONNX_PREFIX_PATH "${CMAKE_BINARY_DIR}/external_projects")

    # Protobuf (required for ONNX)
    ExternalProject_Add(protobuf-external
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
        GIT_SHALLOW ON
        GIT_TAG v31.1
        PREFIX "${CMAKE_BINARY_DIR}/external_projects"
        UPDATE_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_CXX_STANDARD=17
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_ABSL_PROVIDER=module
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    add_library(protobuf::libprotobuf STATIC IMPORTED)
    add_dependencies(protobuf::libprotobuf protobuf-external)

    # utf8_range (protobuf dependency)
    add_library(utf8_range::utf8_range STATIC IMPORTED)
    set_target_properties(utf8_range::utf8_range PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}utf8_range${CMAKE_STATIC_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
    target_link_libraries(protobuf::libprotobuf INTERFACE utf8_range::utf8_range)

    # Setup abseil library dependencies
    include(${CMAKE_SOURCE_DIR}/cmake/abseil_dependencies.cmake)
    setup_abseil_dependencies(protobuf::libprotobuf)

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/external_projects/include")
    set_target_properties(protobuf::libprotobuf PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
endif()

# ONNX - built from source in Case 2 and Case 3
if(NOT USE_SYSTEM_ONNX)
    set(ONNX_EXTERNAL_DEPENDS "")
    if(NOT USE_SYSTEM_PROTOBUF)
        set(ONNX_EXTERNAL_DEPENDS DEPENDS protobuf-external)
    endif()

    ExternalProject_Add(onnx-external
        ${ONNX_EXTERNAL_DEPENDS}
        GIT_REPOSITORY https://github.com/onnx/onnx
        GIT_SHALLOW ON
        GIT_TAG v1.19.1
        PREFIX "${CMAKE_BINARY_DIR}/external_projects"
        UPDATE_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_CXX_STANDARD=17
            -DONNX_BUILD_TESTS=OFF
            -DCMAKE_PREFIX_PATH=${ONNX_PREFIX_PATH}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    set(ONNX_LIB_PATH "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(ONNX_PROTO_LIB_PATH "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx_proto${CMAKE_STATIC_LIBRARY_SUFFIX}")

    add_library(ONNX::onnx STATIC IMPORTED)
    add_library(ONNX::onnx_proto STATIC IMPORTED)
    add_dependencies(ONNX::onnx onnx-external)
    add_dependencies(ONNX::onnx_proto onnx-external)
    target_link_libraries(ONNX::onnx INTERFACE protobuf::libprotobuf)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/external_projects/include")
    set_target_properties(ONNX::onnx PROPERTIES
        IMPORTED_LOCATION "${ONNX_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
    set_target_properties(ONNX::onnx_proto PROPERTIES
        IMPORTED_LOCATION "${ONNX_PROTO_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
endif()

# Shared library that bundles ONNX dependencies
# The onnx_wrapper.cpp file instantiates ONNX types to pull in the needed symbols
add_library(onnx-wrapper SHARED
    src/onnx_wrapper.cpp
)

# Make sure library depends on external projects (only when building from source)
if(NOT USE_SYSTEM_ONNX)
    if(USE_SYSTEM_PROTOBUF)
        # Only ONNX built from source
        add_dependencies(onnx-wrapper onnx-external)
    else()
        # Both built from source
        add_dependencies(onnx-wrapper protobuf-external onnx-external)
    endif()
endif()

# Link ONNX libraries - the symbol usage in onnx_wrapper.cpp pulls in what's needed
target_link_libraries(onnx-wrapper
    PRIVATE
        ONNX::onnx
        ONNX::onnx_proto
        protobuf::libprotobuf
)

if(NOT USE_SYSTEM_ONNX)
    target_include_directories(onnx-wrapper
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/external_projects/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

# Create namespaced alias
add_library(onnx-wrapper::onnx-wrapper ALIAS onnx-wrapper)

# Install targets
install(TARGETS onnx-wrapper
    EXPORT onnx-wrapper-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(NOT USE_SYSTEM_ONNX)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/external_projects/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(EXPORT onnx-wrapper-targets
    FILE onnx-wrapper-targets.cmake
    NAMESPACE onnx-wrapper::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onnx-wrapper
)

# Generate package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/onnx-wrapper-config.cmake.in
    ${CMAKE_BINARY_DIR}/onnx-wrapper-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onnx-wrapper
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/onnx-wrapper-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/onnx-wrapper-config.cmake
    ${CMAKE_BINARY_DIR}/onnx-wrapper-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onnx-wrapper
)
