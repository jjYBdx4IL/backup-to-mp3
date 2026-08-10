if(NOT EXISTS "${ARCHIVE}")
	message(FATAL_ERROR "extract_fldigi_archive.cmake: ${ARCHIVE} does not exist")
endif()

if(EXISTS "${DEST}")
	file(REMOVE_RECURSE "${DEST}")
endif()

set(_stage "${DEST}.extract-stage")
if(EXISTS "${_stage}")
	file(REMOVE_RECURSE "${_stage}")
endif()
file(MAKE_DIRECTORY "${_stage}")

file(ARCHIVE_EXTRACT INPUT "${ARCHIVE}" DESTINATION "${_stage}")

file(GLOB _stage_children LIST_DIRECTORIES true "${_stage}/*")
list(LENGTH _stage_children _n)
if(NOT _n EQUAL 1 OR NOT IS_DIRECTORY "${_stage_children}")
	message(FATAL_ERROR
		"extract_fldigi_archive.cmake: expected exactly one top-level directory in "
		"${ARCHIVE}, found: ${_stage_children}"
	)
endif()

file(RENAME "${_stage_children}" "${DEST}")
file(REMOVE_RECURSE "${_stage}")
