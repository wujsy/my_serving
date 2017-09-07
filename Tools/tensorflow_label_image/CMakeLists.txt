cmake_minimum_required(VERSION 2.8.7)

set( CMAKE_VERBOSE_MAKEFILE off )
set( CMAKE_BUILD_TYPE Release )
set(CMAKE_CXX_STANDARD 14)

project(label_image C CXX)
set(IMAGE_SRCS cv_process.cpp  main.cc)
add_executable(label_image ${IMAGE_SRCS}) 

if(MSVC)
    set(TENSORFLOW_FOLDER c:/Users/Administrator/Desktop/tensorflow)
else()
    set(TENSORFLOW_FOLDER /data/pzl/tensorflow)
endif()

# Add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include_directories(${TENSORFLOW_FOLDER})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
endif()

if(MSVC)
    include("c:/Users/Administrator/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/caffe-builder-config.cmake")

    include_directories(
        "${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build"
        "${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/external/eigen_archive"
        "${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/external/jpeg_archive"
        "${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/external/protobuf/src/"

        )

    target_link_libraries(label_image 
        ${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/Release/libtensorflow.lib
        ${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/Release/tf_protos_cc.lib
        ${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/protobuf/src/protobuf/Release/libprotoc.lib
        ${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/protobuf/src/protobuf/Release/libprotobuf-lite.lib
        ${TENSORFLOW_FOLDER}/tensorflow/contrib/cmake/build/protobuf/src/protobuf/Release/libprotobuf.lib
        )
else()

    include_directories (
        /usr/local/include
        /usr/include
        /usr/include/openblas
        /usr/include/python2.7
        /usr/local/cuda/include
        )
    include_directories(
        "${TENSORFLOW_FOLDER}/output/output/external/eigen_archive"
        "${TENSORFLOW_FOLDER}/output/output/external/protobuf/src/"
        "${TENSORFLOW_FOLDER}/output/output/execroot/tensorflow/bazel-out/local_linux-opt/genfiles/external/org_tensorflow"
        "${TENSORFLOW_FOLDER}/output/output/execroot/tensorflow/bazel-out/local-opt/genfiles"
        "${TENSORFLOW_FOLDER}/output/output/execroot/tensorflow/bazel-out/local_linux-opt/genfiles"

        )

    target_link_libraries(label_image 
        ${TENSORFLOW_FOLDER}/bazel-bin/tensorflow/libtensorflow_all.so
        "${TENSORFLOW_FOLDER}/output/output/execroot/tensorflow/bazel-out/local_linux-opt/bin/external/jpeg/libsimd_x86_64.a"
        "${TENSORFLOW_FOLDER}/output/output/execroot/tensorflow/bazel-out/local_linux-opt/bin/external/jpeg/libjpeg.a"
        )
endif()

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()


#include(Protobuf)
#include(Eigen)

# Eigen
#find_package(Eigen REQUIRED)
#include_directories(${Eigen_INCLUDE_DIRS})
# Protobuf
#find_package(Protobuf REQUIRED)
#include_directories(${Protobuf_INCLUDE_DIRS})
#target_link_libraries(label_image ${Protobuf_LIBRARIES})

