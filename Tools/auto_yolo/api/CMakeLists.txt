file(GLOB_RECURSE api_srcs "${APIDIR}/*.c")

include_directories(${SRCDIR})

foreach(source_file ${api_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
    
  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
if (CUDA_FOUND) 
  cuda_add_executable(${name} ${source_file})
  target_link_libraries(${name} ${LINKER_LIBS} dnl)
else ()  
  add_executable(${name} ${source_file})
  target_link_libraries(${name} ${LINKER_LIBS} dnl)
endif() 

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/api/${folder}")

  # install
  install(TARGETS ${name} DESTINATION bin)

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standardized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/api/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()


