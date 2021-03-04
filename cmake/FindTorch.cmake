# Copyright (c) 2021 Dongkyu Kim (dkkim1005@gmail.com)

# TORCH_FOUND       : True if 'torch.h' and libraries are exist or False
# TORCH_PREFIX      : root directory of pytorch installation
# TORCH_HEADER_DIR  : path of header files
# TORCH_LIBRARY_DIR :  "" libraries
# TORCH_LIBRARIES   : list of pytorch libraries

IF (NOT TORCH_FOUND)
  IF (NOT PYTHON3_INTERPRETER)
    SET (PYTHON3_INTERPRETER "python3")
  ENDIF (NOT PYTHON3_INTERPRETER)

  # check whether torch is installed at the dist-package of PYTHON3_INTERPRETER via pip module
  EXECUTE_PROCESS (COMMAND bash -c "${PYTHON3_INTERPRETER} -m pip list | grep 'torch' | awk '{print $1}'" OUTPUT_VARIABLE CHECK_PIP_CACHE_TORCH)

  IF (${CHECK_PIP_CACHE_TORCH} MATCHES "torch")
    SET (TORCH_FOUND TRUE)

    # get torch info from pip module
    EXECUTE_PROCESS (COMMAND bash -c "${PYTHON3_INTERPRETER} -m pip show torch | grep Location | sed -e 's/^Location: //g' | tr -d '\n'" OUTPUT_VARIABLE TORCH_PREFIX)
    SET (TORCH_HEADER_DIR ${TORCH_PREFIX}/torch/include/torch/csrc/api/include;${TORCH_PREFIX}/torch/include)
    SET (TORCH_LIBRARY_DIR ${TORCH_PREFIX}/torch/lib)
    MESSAGE (STATUS "TORCH_HEADER_DIR: ${TORCH_HEADER_DIR}")

    # check essential header files
    SET (HEADER_FILES "torch.h")
    FOREACH (HEADER_FILE ${HEADER_FILES})
      MESSAGE (STATUS "  Looking for ${HEADER_FILE}")
      FIND_PATH (HEADER_FILE_PATH NAMES ${HEADER_FILE} HINTS "${TORCH_PREFIX}/torch/include/torch/csrc/api/include/torch")
      IF (HEADER_FILE_PATH)
        MESSAGE (STATUS "  Looking for ${HEADER_FILE} - found")
      ELSE ()
        MESSAGE (WORNING "  ${HEADER_FILE} is not detected at the path: ${TORCH_PREFIX}/torch/include/torch/csrc/api/include/torch")
        SET (TORCH_FOUND FALSE)
      ENDIF (HEADER_FILE_PATH)
    ENDFOREACH ()
    MESSAGE (STATUS "TORCH_LIBRARY_DIR: ${TORCH_LIBRARY_DIR}")

    # suffix of the dynamic library 
    IF (APPLE)
      SET (DYLIBSUFFIX dylib)
    ELSEIF (UNIX)
      SET (DYLIBSUFFIX so)
    ENDIF ()

    # check essential shared libraries
    MACRO (CHECK_LIBRARY LIBRARYNAME LIBRARYPATH)
      #MESSAGE (STATUS "  Looking for ${LIBRARYPATH}/lib${LIBRARYNAME}.${DYLIBSUFFIX}")
      MESSAGE (STATUS "  Looking for lib${LIBRARYNAME}.${DYLIBSUFFIX}")
      FIND_LIBRARY (CHECK_LIBRARY NAMES ${LIBRARYNAME} lib${LIBRARYNAME}  HINTS ${LIBRARYPATH})
      IF (CHECK_LIBRARY)
        #MESSAGE (STATUS "  Looking for ${LIBRARYPATH}/lib${LIBRARYNAME}.${DYLIBSUFFIX} - found")
        MESSAGE (STATUS "  Looking for lib${LIBRARYNAME}.${DYLIBSUFFIX} - found")
      ELSE ()
        MESSAGE (WORNING "  lib${LIBRARYNAME}.${DYLIBSUFFIX} is not detected.")
        SET (TORCH_FOUND FALSE)
      ENDIF ()
    ENDMACRO (CHECK_LIBRARY)

    CHECK_LIBRARY(c10 ${TORCH_LIBRARY_DIR})
    IF (CUDA_FOUND)
      CHECK_LIBRARY(cuda ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
      CHECK_LIBRARY(nvrtc ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      CHECK_LIBRARY(nvToolsExt ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      CHECK_LIBRARY(cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      CHECK_LIBRARY(c10_cuda ${TORCH_LIBRARY_DIR})
    ENDIF ()

    # get cflags and libraries
    SET (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_PREFIX}/torch)
    FIND_PACKAGE (Torch REQUIRED)

    #
    #MACRO (LIBRARY_REGISTER LIBRARYNAME LIBRARYPATH)
    #  FIND_LIBRARY (CHECK_LIBRARY NAMES ${LIBRARYNAME} lib${LIBRARYNAME}  HINTS ${LIBRARYPATH})
    #  IF (CHECK_LIBRARY)
    #    MESSAGE (STATUS "  ${LIBRARYPATH}/lib${LIBRARYNAME}.${DYLIBSUFFIX} - found")
    #    SET (TORCH_LIBRARIES ${TORCH_LIBRARIES} ${LIBRARYPATH}/lib${LIBRARYNAME}.${DYLIBSUFFIX})
    #  ELSE ()
    #    MESSAGE (NOTICE "  lib${LIBRARYNAME}.${DYLIBSUFFIX} is not detected.")
    #  ENDIF ()
    #ENDMACRO (LIBRARY_REGISTER)
    #
    #LIBRARY_REGISTER(c10 ${TORCH_LIBRARY_DIR})
    #LIBRARY_REGISTER(torch_cpu ${TORCH_LIBRARY_DIR})
    #IF (CUDA_FOUND)
    #  LIBRARY_REGISTER(torch_cuda ${TORCH_LIBRARY_DIR})
    #  LIBRARY_REGISTER(c10_cuda ${TORCH_LIBRARY_DIR})
    #  LIBRARY_REGISTER(cuda ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
    #  LIBRARY_REGISTER(nvrtc ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    #  LIBRARY_REGISTER(nvToolsExt ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    #  LIBRARY_REGISTER(cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    #ENDIF ()
    #
    #INCLUDE_DIRECTORIES (SYSTEM ${TORCH_HEADER_DIR})
    #LINK_DIRECTORIES(${TORCH_LIBRARY_DIR})

    IF (NOT TORCH_FOUND)
      MESSAGE (FATAL_ERROR "Sorry. Please check whether pytorch has been installed properly before proceeding to the next step.")
    ENDIF (NOT TORCH_FOUND)
  ELSE ()
    MESSAGE (FATAL_ERROR " pytorch does not exist in pip list of default python3 interpreter.\n You can force to employ a custom python3 interpreter that pytorch currently installed, \n via 'cmake /path/to/CMakeLists.txt -DPYTHON3_INTERPRETER=/path/to/your/custom/python3.x'.")
  ENDIF (${CHECK_PIP_CACHE_TORCH} MATCHES "torch")
ENDIF (NOT TORCH_FOUND)
