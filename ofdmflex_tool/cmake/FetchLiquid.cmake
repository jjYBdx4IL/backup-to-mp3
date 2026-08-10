set(LIQUID_VERSION 1.8.0)
set(LIQUID_URL "https://github.com/jgaeddert/liquid-dsp/archive/refs/tags/v${LIQUID_VERSION}.tar.gz")
set(LIQUID_URL_HASH "SHA256=abef8b2ddfd58c0a84ecda4f62158c4824b916144af4a2b07776e1a144d8cda4")

CPMAddPackage(
	NAME liquid
	VERSION ${LIQUID_VERSION}
	URL ${LIQUID_URL}
	URL_HASH ${LIQUID_URL_HASH}
	DOWNLOAD_ONLY YES
	DOWNLOAD_NO_EXTRACT YES
)

set(LIQUID_CPM_ARCHIVE ${liquid_SOURCE_DIR}/v${LIQUID_VERSION}.tar.gz)
set(LIQUID_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/liquid-src)

if(NOT EXISTS ${LIQUID_SRC_DIR}/CMakeLists.txt)
	execute_process(
		COMMAND ${CMAKE_COMMAND}
			-D ARCHIVE=${LIQUID_CPM_ARCHIVE}
			-D DEST=${LIQUID_SRC_DIR}
			-P ${CMAKE_CURRENT_LIST_DIR}/extract_archive.cmake
		RESULT_VARIABLE LIQUID_EXTRACT_RESULT
	)
	if(NOT LIQUID_EXTRACT_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to extract liquid-dsp archive (${LIQUID_CPM_ARCHIVE})")
	endif()

	execute_process(
		COMMAND ${CMAKE_COMMAND}
			-D LIQUID_SRC_DIR=${LIQUID_SRC_DIR}
			-D PATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/liquid-libfec.patch
			-P ${CMAKE_CURRENT_LIST_DIR}/patch_liquid.cmake
		RESULT_VARIABLE LIQUID_PATCH_RESULT
	)
	if(NOT LIQUID_PATCH_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to patch liquid-dsp libfec detection into ${LIQUID_SRC_DIR}")
	endif()
endif()

add_subdirectory(${LIQUID_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/liquid-build EXCLUDE_FROM_ALL)
