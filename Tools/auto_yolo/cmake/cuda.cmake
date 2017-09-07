if(DEFINED USE_CPU)
  return()
endif()
################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

  # dynamic libs have different suffix in mac and linux
  if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib")
  else()
    set(CUDNN_LIB_NAME "libcudnn.so")
  endif()

  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
   PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
   DOC "Path to cuDNN library.")
  
  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    # cuDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "???")
    else()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
    if(cuDNNVersionIncompatible)
      message(FATAL_ERROR "cuDNN version >3 is required.")
    endif()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

  endif()
endfunction()

# ---[ CUDA
FIND_PACKAGE(CUDA)  
if (CUDA_FOUND)
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
    message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND LINKER_LIBS ${CUDA_LIBRARIES}
                            ${CUDA_CUBLAS_LIBRARIES}
                            ${CUDA_curand_LIBRARY}
                            ${CUDA_cusparse_LIBRARY})
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-Xcompiler \"-fPIC\" ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=compute_20 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=compute_30 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=compute_35 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=compute_50 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=compute_60 ")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=compute_61 ")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    add_definitions(-DGPU)    
    cuda_include_directories(src)
else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
endif()

# ---[ CUDNN
detect_cuDNN()
if(HAVE_CUDNN)
    message(STATUS "CUDNN FOUND")
    add_definitions(-DCUDNN)
    list(APPEND LINKER_LIBS ${CUDNN_LIBRARY})
else()
    message(WARNING "-- CUDNN is not detected by cmake. Building without it...")
endif()
