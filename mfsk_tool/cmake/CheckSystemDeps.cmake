include(CheckIncludeFileCXX)
include(FindPkgConfig)

set(_missing_pkgs "")
set(_missing_reasons "")

macro(harness_require_header header apt_package)
	check_include_file_cxx("${header}" HAVE_${apt_package}_HEADER)
	if(NOT HAVE_${apt_package}_HEADER)
		list(APPEND _missing_pkgs "${apt_package}")
		list(APPEND _missing_reasons "  ${apt_package} (missing header: ${header})")
	endif()
	unset(HAVE_${apt_package}_HEADER CACHE)
endmacro()

macro(harness_require_program prog apt_package)
	find_program(_prog_${apt_package} NAMES ${prog})
	if(NOT _prog_${apt_package})
		list(APPEND _missing_pkgs "${apt_package}")
		list(APPEND _missing_reasons "  ${apt_package} (missing program: ${prog})")
	endif()
endmacro()

harness_require_header("portaudio.h" "portaudio19-dev")
harness_require_header("sndfile.h" "libsndfile1-dev")
harness_require_header("samplerate.h" "libsamplerate0-dev")
harness_require_header("png.h" "libpng-dev")
harness_require_header("X11/Xlib.h" "libx11-dev")
harness_require_header("alsa/asoundlib.h" "libasound2-dev")
harness_require_header("lame/lame.h" "libmp3lame-dev")
harness_require_header("mpg123.h" "libmpg123-dev")

harness_require_program("patch" "patch")
harness_require_program("make;gmake" "build-essential")

find_program(HARNESS_FLTK_CONFIG NAMES fltk-config)
if(NOT HARNESS_FLTK_CONFIG)
	list(APPEND _missing_pkgs "libfltk1.3-dev")
	list(APPEND _missing_reasons "  libfltk1.3-dev (missing program: fltk-config)")
endif()

if(_missing_pkgs)
	list(REMOVE_DUPLICATES _missing_pkgs)
	list(JOIN _missing_reasons "\n" _reasons_joined)
	string(REPLACE ";" " " _pkgs_joined "${_missing_pkgs}")
	message(FATAL_ERROR
		"Missing distro packages required to build the harness:\n"
		"${_reasons_joined}\n"
		"\n"
		"Quickest fix (installs everything fldigi itself needs to build, a superset "
		"of the above):\n"
		"  sudo apt-get build-dep fldigi\n"
		"\n"
		"Or install just what's missing:\n"
		"  sudo apt-get install ${_pkgs_joined}\n"
	)
endif()

pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
pkg_check_modules(ALSA REQUIRED IMPORTED_TARGET alsa)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SAMPLERATE REQUIRED IMPORTED_TARGET samplerate)
pkg_check_modules(PNG REQUIRED IMPORTED_TARGET libpng16)
pkg_check_modules(X11 REQUIRED IMPORTED_TARGET x11)
pkg_check_modules(MPG123 REQUIRED IMPORTED_TARGET libmpg123)

# libmp3lame ships no .pc file on Debian/Ubuntu - locate it by hand.
find_library(LAME_LIBRARY NAMES mp3lame)
find_path(LAME_INCLUDE_DIR NAMES lame/lame.h)
if(NOT LAME_LIBRARY OR NOT LAME_INCLUDE_DIR)
	message(FATAL_ERROR "libmp3lame not found - install libmp3lame-dev")
endif()
add_library(mp3lame_imported UNKNOWN IMPORTED)
set_target_properties(mp3lame_imported PROPERTIES
	IMPORTED_LOCATION "${LAME_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${LAME_INCLUDE_DIR}")
