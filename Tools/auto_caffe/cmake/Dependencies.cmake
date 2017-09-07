# These lists are later turned into target properties on main caffe library target
set(Caffe_LINKER_LIBS "")
set(Caffe_INCLUDE_DIRS "")
set(Caffe_DEFINITIONS "")
set(Caffe_COMPILE_OPTIONS "")

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if(USE_OPENMP)
  # Ideally, this should be provided by the BLAS library IMPORTED target. However,
  # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
  # to flick the switch manually as needed.
  #
  # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
  # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
  # However, this naïve method will force any user of Caffe to add the same kludge
  # into their buildsystem again, so we put these options into per-target PUBLIC
  # compile options and link flags, so that they will be exported properly.
  find_package(OpenMP REQUIRED)
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Caffe_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---[ Google-glog
include("cmake/External/glog.cmake")
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${Snappy_LIBRARIES})
endif()

# ---[ CUDA
if(NOT DEFINED USE_CPU)
  include(cmake/Cuda.cmake)
endif()

if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
  endif()

  list(APPEND Caffe_DEFINITIONS PUBLIC -DCPU_ONLY)
endif()

# ---[ CUB
if(HAVE_CUDA)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC "/data1/workspace/online_serving/Tools/cub-1.6.4")
endif()

if(USE_NCCL)
  find_package(NCCL REQUIRED)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARIES})
  add_definitions(-DUSE_NCCL)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_OPENCV)
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "open" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
    list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_ACCELERATE)
    endif()
  endif()
endif()

