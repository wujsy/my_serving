# These lists are later turned into target properties on main caffe library target
set(LINKER_LIBS "")

# ---[ thread
FIND_PACKAGE(Threads)
list(APPEND LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${OpenCV_LIBS})
add_definitions(-DOPENCV)

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${Boost_LIBRARIES})

# ---[ CUDA
include(cmake/cuda.cmake)
