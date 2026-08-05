include(ExternalProject)

set(FLDIGI_VERSION 4.2.13)
set(FLDIGI_URL "https://sourceforge.net/projects/fldigi/files/fldigi/fldigi-${FLDIGI_VERSION}.tar.gz/download")
set(FLDIGI_URL_HASH "SHA256=a1e8d990359ce9c0cce3ceb5116fd0cf72c95528969766898640ca6ca2dba8d4")

CPMAddPackage(
	NAME fldigi
	VERSION ${FLDIGI_VERSION}
	URL ${FLDIGI_URL}
	URL_HASH ${FLDIGI_URL_HASH}
	DOWNLOAD_ONLY YES
	DOWNLOAD_NO_EXTRACT YES
)

set(FLDIGI_CPM_ARCHIVE ${fldigi_SOURCE_DIR}/fldigi-${FLDIGI_VERSION}.tar.gz)

set(FLDIGI_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/fldigi-src)

set(FLDIGI_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/fldigi-build)

set(FLDIGI_OBJLIST
	trx/fldigi-modem.o trx/fldigi-trx.o
	mfsk/fldigi-interleave.o mfsk/fldigi-mfsk.o mfsk/fldigi-mfskvaricode.o
	rtty/fldigi-fsk.o rtty/fldigi-rtty.o
	filters/fldigi-fftfilt.o filters/fldigi-filters.o filters/fldigi-viterbi.o
	soundcard/fldigi-sound.o soundcard/fldigi-soundconf.o
	globals/fldigi-globals.o
	misc/fldigi-ascii.o misc/fldigi-strutil.o misc/fldigi-threads.o
	misc/fldigi-timeops.o misc/fldigi-util.o misc/fldigi-log.o misc/fldigi-misc.o
	debug/fldigi-debug.o
	cw/fldigi-morse.o
	widgets/fldigi-plot_xy.o widgets/fldigi-picture.o
	rsid/fldigi-rsid.o
	widgets/fldigi-flinput2.o
)

set(FLDIGI_OBJ_ABS_PATHS "")
foreach(_obj ${FLDIGI_OBJLIST})
	list(APPEND FLDIGI_OBJ_ABS_PATHS "${FLDIGI_BUILD_DIR}/src/${_obj}")
endforeach()

set(FLDIGI_ARCHIVE "${FLDIGI_BUILD_DIR}/src/libfldigi_harness_objs.a")

include(ProcessorCount)
ProcessorCount(FLDIGI_NPROC)
if(FLDIGI_NPROC EQUAL 0)
	set(FLDIGI_NPROC 1)
endif()

find_program(FLDIGI_MAKE_EXECUTABLE NAMES make gmake REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(FLDIGI_CXXFLAGS "-O2 -DNDEBUG -fno-delete-null-pointer-checks")
else()
	set(FLDIGI_CXXFLAGS "-g -O0 -fno-delete-null-pointer-checks")
endif()

ExternalProject_Add(fldigi_build
	SOURCE_DIR ${FLDIGI_SRC_DIR}
	BINARY_DIR ${FLDIGI_BUILD_DIR}
	DOWNLOAD_COMMAND ${CMAKE_COMMAND}
		-D ARCHIVE=${FLDIGI_CPM_ARCHIVE}
		-D DEST=${FLDIGI_SRC_DIR}
		-P ${CMAKE_CURRENT_LIST_DIR}/extract_fldigi_archive.cmake
	UPDATE_COMMAND ""
	PATCH_COMMAND ${CMAKE_COMMAND}
		-D FLDIGI_SRC_DIR=${FLDIGI_SRC_DIR}
		-D PATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/waterfall-null-guard.patch
		-P ${CMAKE_CURRENT_LIST_DIR}/patch_fldigi.cmake
	CONFIGURE_COMMAND ${FLDIGI_SRC_DIR}/configure --without-pulseaudio
	BUILD_COMMAND ${FLDIGI_MAKE_EXECUTABLE} -C ${FLDIGI_BUILD_DIR}/src -j${FLDIGI_NPROC}
		"CXXFLAGS=${FLDIGI_CXXFLAGS}"
		${FLDIGI_OBJLIST}
	COMMAND ${CMAKE_AR} rcs ${FLDIGI_ARCHIVE} ${FLDIGI_OBJ_ABS_PATHS}
	BUILD_BYPRODUCTS ${FLDIGI_OBJ_ABS_PATHS} ${FLDIGI_ARCHIVE}
	INSTALL_COMMAND ""
)

add_library(fldigi_harness_objs STATIC IMPORTED GLOBAL)
set_target_properties(fldigi_harness_objs PROPERTIES IMPORTED_LOCATION ${FLDIGI_ARCHIVE})
add_dependencies(fldigi_harness_objs fldigi_build)

add_library(fldigi_headers INTERFACE)
target_include_directories(fldigi_headers INTERFACE
	${FLDIGI_BUILD_DIR}/src
	${FLDIGI_SRC_DIR}/src
	${FLDIGI_SRC_DIR}/src/include
	${FLDIGI_SRC_DIR}/src/irrxml
	${FLDIGI_SRC_DIR}/src/libtiniconv
	${FLDIGI_SRC_DIR}/src/fileselector
	${FLDIGI_SRC_DIR}/src/xmlrpcpp
	${FLDIGI_SRC_DIR}/src/mbedtls
)
