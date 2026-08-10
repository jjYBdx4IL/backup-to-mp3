set(_target "${LIQUID_SRC_DIR}/CMakeLists.txt")
set(_marker "HAVE_LIBFEC")

if(NOT EXISTS "${_target}")
	message(FATAL_ERROR "patch_liquid.cmake: ${_target} does not exist")
endif()

file(READ "${_target}" _contents)
string(FIND "${_contents}" "${_marker}" _pos)

if(_pos EQUAL -1)
	message(STATUS "Applying ${PATCH_FILE} to ${_target}")
	execute_process(
		COMMAND patch -p1 --forward -i "${PATCH_FILE}"
		WORKING_DIRECTORY "${LIQUID_SRC_DIR}"
		RESULT_VARIABLE _rc
	)
	if(NOT _rc EQUAL 0)
		message(FATAL_ERROR "Failed to apply ${PATCH_FILE} (patch exited ${_rc})")
	endif()
else()
	message(STATUS "liquid-dsp libfec-detection patch already applied, skipping")
endif()
