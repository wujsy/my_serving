FILE(GLOB SRC_FILES ${SRCDIR}/*.c)
FILE(GLOB CU_SRC_FILES ${SRCDIR}/*.cu)
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/darknet.c")

if (CUDA_FOUND)
  message(STATUS "CUDA FOUND")
  cuda_add_library(dnl SHARED ${SRC_FILES} ${CU_SRC_FILES})
  target_link_libraries(dnl ${LINKER_LIBS})
  cuda_add_executable(darknet ${SRCDIR}/darknet.c)
  target_link_libraries(darknet ${LINKER_LIBS} dnl)
else ()
  add_library(dnl SHARED ${SRC_FILES} ${CU_SRC_FILES})
  target_link_libraries(dnl ${LINKER_LIBS})
  add_executable(darknet ${SRCDIR}/darknet.c)
  target_link_libraries(darknet ${LINKER_LIBS} dnl)
endif()

SET(LINKER_LIBS dnl ${LINKER_LIBS} PARENT_SCOPE)
