############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cs344)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/gcc; 
  -gencode;arch=compute_80,code=sm_80;  
  -gencode;arch=compute_90,code=sm_90;")
  #-gencode;arch=compute_20,code=sm_20; 
  #-gencode;arch=compute_11,code=sm_11; 
  #-gencode;arch=compute_12,code=sm_12;
  #-gencode;arch=compute_13,code=sm_13;")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	#set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory ("Problem_Sets/Problem_Set_1")
  add_subdirectory ("Problem_Sets/Problem_Set_2")
  add_subdirectory ("Problem_Sets/Problem_Set_3")
  add_subdirectory ("Problem_Sets/Problem_Set_4")
  add_subdirectory ("Problem_Sets/Problem_Set_5")
  add_subdirectory ("Problem_Sets/Problem_Set_6")
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
