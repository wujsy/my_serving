cmake_minimum_required(VERSION 2.8)

set( CMAKE_VERBOSE_MAKEFILE off )
set( CMAKE_BUILD_TYPE Release )
#set( CMAKE_BUILD_TYPE Debug )
PROJECT(image_recognition)
#set(SRC_LIST  main.cc)
#set(SRC_LIST label_image.cc)
set(SRC_LIST object_detection.cc)


set(INC_DIR ./include)  
set(LINK_DIR ./lib)
set(TF_DIR /data/wujiaxing/workspace/tf_build/tensorflow/output)

## Compiler settings
set(CMAKE_CXX_FLAGS "-std=c++0x -Wl,--export-dynamic -Wall -Wno-sign-compare -fPIC")
if(NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Release or Debug" )
endif()

ADD_EXECUTABLE(image_recognition ${SRC_LIST})
set_target_properties(image_recognition PROPERTIES LINKER_LANGUAGE CXX)

include_directories(
	${INC_DIR}/tensorflow
	${INC_DIR}/eigen_archive
        ${INC_DIR}/genfiles
        ${INC_DIR}/org_tensorflow
	${INC_DIR}/protobuf/src
	)

#include_directories(${INC_DIR}/third_party/protobuf/src)
#include_directories(
	#"${TF_DIR}/execroot/org_tensorflow"
	#"${TF_DIR}/execroot/tensorflow/bazel-out/local_linux-opt/genfiles"	
#	)

#include_directories (
#        /usr/local/include
#        /usr/include
#        /usr/include/openblas
#        /usr/include/python2.7
#        /usr/local/cuda/include
#        )

link_directories(${LINK_DIR})
target_link_libraries(image_recognition 
        "/data/wjx/workspace/git/tensorflow_pro/lib/libtensorflow_all.so"
        #"/data/wujiaxing/test/tensorflow_pro/lib/libtensorflow_all_cpu.so"
	#"${TF_DIR}/execroot/tensorflow/bazel-out/local_linux-opt/bin/external/jpeg/libsimd_x86_64.a"
        #"${TF_DIR}/execroot/tensorflow/bazel-out/local_linux-opt/bin/external/jpeg/libjpeg.a"
        )

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()


# Eigen
#include(Eigen)
#add_dependencies(<image_recognition> Eigen)

# Protobuf
#include(Protobuf)
#add_dependencies(<image_recognition> Protobuf)



