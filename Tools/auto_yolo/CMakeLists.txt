cmake_minimum_required(VERSION 2.8)

set( DarkNet_VERSION_MAJOR 1 )
set( DarkNet_VERSION_MINOR 0 )
set( DarkNet_VERSION_DETAIL 0 )

PROJECT( "DarkNet-${DarkNet_VERSION_MAJOR}.${DarkNet_VERSION_MINOR}" )

## Compiler settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--export-dynamic -Wall -Wno-sign-compare -fPIC")

if(NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Release or Debug" )
endif()

set(CMAKE_VERBOSE_MAKEFILE on)
set(THISDIR ${CMAKE_SOURCE_DIR})
set(SRCDIR ${THISDIR}/src)
set(BINDIR ${THISDIR}/bin)
set(LIBDIR ${THISDIR}/lib)
set(APIDIR ${THISDIR}/api)
set(CMKDIR ${THISDIR}/cmake)
set(DOCDIR ${THISDIR}/doc)
set(BDSEISFWI ${THISDIR})
set(EXECUTABLE_OUTPUT_PATH ${BINDIR})
set(LIBRARY_OUTPUT_PATH ${LIBDIR})

# ---[ Dependencies
include(${CMKDIR}/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(api)



