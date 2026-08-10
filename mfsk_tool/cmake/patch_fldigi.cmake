set(_target "${FLDIGI_SRC_DIR}/src/include/waterfall.h")
set(_marker "if (!this) return;")

if(NOT EXISTS "${_target}")
	message(FATAL_ERROR "patch_fldigi.cmake: ${_target} does not exist")
endif()

file(READ "${_target}" _contents)
string(FIND "${_contents}" "${_marker}" _pos)

if(_pos EQUAL -1)
	message(STATUS "Applying ${PATCH_FILE} to ${_target}")
	execute_process(
		COMMAND patch -p1 --forward -i "${PATCH_FILE}"
		WORKING_DIRECTORY "${FLDIGI_SRC_DIR}"
		RESULT_VARIABLE _rc
	)
	if(NOT _rc EQUAL 0)
		message(FATAL_ERROR "Failed to apply ${PATCH_FILE} (patch exited ${_rc})")
	endif()
else()
	message(STATUS "waterfall.h null-guard patch already applied, skipping")
endif()
