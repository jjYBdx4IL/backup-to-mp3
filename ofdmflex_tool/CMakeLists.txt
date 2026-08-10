cmake_minimum_required(VERSION 3.19)
project(ofdmflex_tool C)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	execute_process(COMMAND id -u OUTPUT_VARIABLE OFDMFLEX_UID OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(OFDMFLEX_UID EQUAL 0)
		set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install path prefix" FORCE)
	else()
		set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install path prefix" FORCE)
	endif()
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CTest)
include(cmake/get_cpm.cmake)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_AUTOTESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)

include(cmake/FetchLiquid.cmake)

find_package(OpenSSL REQUIRED)

find_path(MP3LAME_INCLUDE_DIR NAMES lame/lame.h)
find_library(MP3LAME_LIBRARY NAMES mp3lame)
if(NOT MP3LAME_INCLUDE_DIR OR NOT MP3LAME_LIBRARY)
	message(FATAL_ERROR "libmp3lame not found -- install libmp3lame-dev (Debian/Ubuntu) or lame-devel (Fedora)")
endif()
add_library(MP3lame::MP3lame UNKNOWN IMPORTED)
set_target_properties(MP3lame::MP3lame PROPERTIES
	IMPORTED_LOCATION "${MP3LAME_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${MP3LAME_INCLUDE_DIR}"
)

add_executable(ofdmflex_tool src/main.c src/mp3.c)
target_link_libraries(ofdmflex_tool PRIVATE liquid::liquid-static m OpenSSL::Crypto MP3lame::MP3lame)

add_executable(channel_wav src/channel_wav.c)
target_link_libraries(channel_wav PRIVATE liquid::liquid-static m)

install(TARGETS ofdmflex_tool RUNTIME DESTINATION bin)

if(BUILD_TESTING)
	# Digital round-trip smoke test: random data, through the channel-noise
	# stage, encode/decode must recover it byte-exact.
	add_test(NAME roundtrip_smoke
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --channel --channel-awgn
	)
	add_test(NAME roundtrip_smoke_ask2
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --channel --channel-awgn --ofdm-params "-m ask2 -M 20"
	)
	add_test(NAME roundtrip_smoke_ask2_clean
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --ofdm-params "-m ask2 -M 20"
	)
	add_test(NAME roundtrip_smoke_ask2_ir_only
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --ofdm-params "-m ask2 -M 20" --ir
	)
	add_test(NAME roundtrip_smoke_ask2_full_channel
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --ofdm-params "-m ask2" --ir --channel --channel-awgn
	)
	add_test(NAME roundtrip_smoke_ask4_full_channel
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh -n 4096 --ofdm-params "-m ask4" --ir --channel --channel-awgn
	)
	set_tests_properties(
		roundtrip_smoke
		roundtrip_smoke_ask2
		roundtrip_smoke_ask2_clean
		roundtrip_smoke_ask2_ir_only
		roundtrip_smoke_ask2_full_channel
		roundtrip_smoke_ask4_full_channel
		PROPERTIES
		ENVIRONMENT "OFDMFLEX_TOOL_BIN=$<TARGET_FILE:ofdmflex_tool>;CHANNEL_WAV_BIN=$<TARGET_FILE:channel_wav>;IR_FILE=${CMAKE_CURRENT_SOURCE_DIR}/tests/ir-48k.wav"
		TIMEOUT 120
	)

	# Real-channel replica: ir-custom-channel-48k.wav is a genuine impulse
	# response measured off a live custom channel (via channel_probe.py's
	# log-sweep deconvolution -- see tests/extract_channel_ir.py to
	# regenerate it from a fresh capture), combined with that same
	# capture's measured receiver noise floor and TX/RX clock drift. The
	# encode parameters are the ones found (by an offline sweep against
	# this same replica, then confirmed over the live channel) to be the
	# fastest settings that still decode reliably: default parameters
	# (-C 16 cyclic prefix, default carrier placement) fail outright
	# against this channel's ~8ms multipath delay spread and bandpass
	# shape, which is exactly what this test guards against regressing.
	add_test(NAME roundtrip_custom_channel
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh
			--ofdm-params "--preset phone1"
			-n 16000 --mp3
			--drift-ppm -107.111 --ir --ir-preserve-gain --post-ir-noise-floor -77.05
	)
	add_test(NAME roundtrip_custom_channel_mp3_internal
		COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh
			--ofdm-params "--preset phone1"
			-n 16000 --mp3 --mp3-engine internal
	)
	set_tests_properties(roundtrip_custom_channel roundtrip_custom_channel_mp3_internal PROPERTIES
		ENVIRONMENT "OFDMFLEX_TOOL_BIN=$<TARGET_FILE:ofdmflex_tool>;CHANNEL_WAV_BIN=$<TARGET_FILE:channel_wav>;IR_FILE=${CMAKE_CURRENT_SOURCE_DIR}/tests/ir-custom-channel-48k.wav"
		TIMEOUT 60
	)
endif()
