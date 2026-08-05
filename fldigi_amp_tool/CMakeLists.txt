cmake_minimum_required(VERSION 3.19)
project(fldigi_amp_harness CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # gnu++17, matching harness/build/cxxflags.txt

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug) # harness's own sources default to -g -O0; pass --config Release for -O2
endif()

set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CTest)

include(cmake/CheckOS.cmake)
include(cmake/CheckSystemDeps.cmake)
include(cmake/get_cpm.cmake)
include(cmake/FetchFldigi.cmake)

find_package(Threads REQUIRED)

execute_process(
	COMMAND fltk-config --cxxflags
	OUTPUT_VARIABLE FLTK_CXXFLAGS_RAW
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(FLTK_CXXFLAGS UNIX_COMMAND "${FLTK_CXXFLAGS_RAW}")

execute_process(
	COMMAND fltk-config --use-images --ldflags
	OUTPUT_VARIABLE FLTK_LDFLAGS_RAW
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(FLTK_LDFLAGS UNIX_COMMAND "${FLTK_LDFLAGS_RAW}")

pkg_check_modules(OPUS QUIET opus)
pkg_check_modules(LIBUSB QUIET libusb-1.0)

add_library(harness_compile_settings INTERFACE)
target_compile_definitions(harness_compile_settings INTERFACE HAVE_CONFIG_H BUILD_FLDIGI)
target_compile_options(harness_compile_settings INTERFACE
	-pthread ${FLTK_CXXFLAGS}
	$<$<CONFIG:Debug>:-g -O0>
	$<$<CONFIG:Release>:-O2 -DNDEBUG>
)
target_include_directories(harness_compile_settings INTERFACE ${OPUS_INCLUDE_DIRS} ${LIBUSB_INCLUDE_DIRS})
target_link_libraries(harness_compile_settings INTERFACE
	fldigi_headers
	Threads::Threads
	${FLTK_LDFLAGS}
	PkgConfig::PORTAUDIO
	PkgConfig::ALSA
	PkgConfig::SNDFILE
	PkgConfig::SAMPLERATE
	PkgConfig::PNG
	PkgConfig::X11
	m
)

add_library(harness_stubs STATIC src/stubs.cxx src/harness_io.cxx)
target_link_libraries(harness_stubs PUBLIC harness_compile_settings)
add_dependencies(harness_stubs fldigi_build)

add_library(harness_amp_proto STATIC src/amp_proto.cxx)
target_link_libraries(harness_amp_proto PUBLIC harness_compile_settings)

add_executable(fldigi_amp_tool src/main.cxx)
target_link_libraries(fldigi_amp_tool PRIVATE
	harness_stubs harness_amp_proto fldigi_harness_objs harness_compile_settings
)

add_executable(analyze_repeats src/analyze_repeats.cxx)
target_link_libraries(analyze_repeats PRIVATE harness_amp_proto harness_compile_settings)

add_executable(bitrate_sweep src/bitrate_sweep.cxx)
target_link_libraries(bitrate_sweep PRIVATE
	harness_stubs fldigi_harness_objs harness_compile_settings
)

add_executable(rsid_check src/rsid_check.cxx)
target_link_libraries(rsid_check PRIVATE
	harness_stubs fldigi_harness_objs harness_compile_settings
)

if(BUILD_TESTING)
	# Digital (no airgap simulation) round-trip — REQUIREMENTS.md verification #1a.
	add_test(NAME roundtrip_small
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip_test.sh
			$<TARGET_FILE:fldigi_amp_tool>
			${CMAKE_CURRENT_SOURCE_DIR}/testdata/small.txt
			${CMAKE_CURRENT_BINARY_DIR}/test-tmp/roundtrip_small
	)

	# Same, but against a ~1KB tar.gz archive generated on the fly from one of the
	# harness's own source files (see tests/roundtrip_generated_archive_test.sh)
	# instead of a checked-in fixture — closer in size/shape to the real
	# 7z-archive use case than roundtrip_small's 85-byte text file. Encode+decode
	# of ~2.5 minutes of 8kHz audio is well under a minute at full CPU speed, but
	# give it room.
	add_test(NAME roundtrip_generated_archive
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip_generated_archive_test.sh
			$<TARGET_FILE:fldigi_amp_tool>
			${CMAKE_CURRENT_SOURCE_DIR}/src
			${CMAKE_CURRENT_BINARY_DIR}/test-tmp/roundtrip_generated_archive
	)
	set_tests_properties(roundtrip_generated_archive PROPERTIES TIMEOUT 300)

	# rsid_check smoke test: confirm it runs cleanly over a real encoded WAV.
	add_test(NAME rsid_check_smoke
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/rsid_check_test.sh
			$<TARGET_FILE:fldigi_amp_tool>
			$<TARGET_FILE:rsid_check>
			${CMAKE_CURRENT_SOURCE_DIR}/testdata/small.txt
			${CMAKE_CURRENT_BINARY_DIR}/test-tmp/rsid_check_smoke
	)

	# analyze_repeats smoke test against the recorded dump fixtures (task #2 in
	# SUMMARY.md) — just confirms it parses real captured header/payload text
	# without crashing.
	add_test(NAME analyze_repeats_smoke
		COMMAND analyze_repeats
			${CMAKE_CURRENT_SOURCE_DIR}/testdata/dump/hdr_text.txt
			${CMAKE_CURRENT_SOURCE_DIR}/testdata/dump/mfsk_text.txt
	)
endif()
