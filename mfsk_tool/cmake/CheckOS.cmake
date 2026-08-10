set(HARNESS_EXPECTED_OS_ID "debian")
set(HARNESS_EXPECTED_OS_VERSION_ID "13")

set(_os_id "")
set(_os_version_id "")
if(EXISTS "/etc/os-release")
	file(STRINGS "/etc/os-release" _os_release_lines)
	foreach(_line ${_os_release_lines})
		if(_line MATCHES "^ID=\"?([^\"]*)\"?$")
			set(_os_id "${CMAKE_MATCH_1}")
		elseif(_line MATCHES "^VERSION_ID=\"?([^\"]*)\"?$")
			set(_os_version_id "${CMAKE_MATCH_1}")
		endif()
	endforeach()
endif()

if(NOT _os_id STREQUAL HARNESS_EXPECTED_OS_ID OR NOT _os_version_id STREQUAL HARNESS_EXPECTED_OS_VERSION_ID)
	if(_os_id STREQUAL "")
		set(_detected "unknown (no /etc/os-release)")
	else()
		set(_detected "${_os_id} ${_os_version_id}")
	endif()
	message(WARNING
		"This build was set up and verified on Debian 13 (trixie). "
		"Detected OS: ${_detected}. Package names, fldigi's ./configure behavior, "
		"and the compiler flags this recipe hardcodes (e.g. "
		"-fno-delete-null-pointer-checks for GCC) may not carry over to a different "
		"distro/version. Proceeding anyway, but if the build fails, this is the "
		"first thing to suspect."
	)
endif()
