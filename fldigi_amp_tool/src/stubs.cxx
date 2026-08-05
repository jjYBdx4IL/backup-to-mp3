// SPDX-License-Identifier: GPL-3.0-or-later
// stubs.cxx
//
// Headless replacement for fldigi's FLTK GUI shell (normally provided by
// fl_digi.cxx, main.cxx, confdialog.cxx, the logbook/, dxcluster/,
// rigcontrol/, waterfall/, dialogs/ subsystems, ...).
//
// We link the REAL fldigi DSP/modem object code (trx, modem, mfsk, rtty,
// filters, viterbi, interleave, sound, rsid, morse, ...) completely
// unmodified, and this translation unit supplies everything that code
// expects from "the rest of the application" so no FLTK window is ever
// created or shown, and no GUI/rig-control/serial-port/network subsystem
// is pulled in.
//
// Two seams matter functionally:
//   - get_tx_char() / put_rx_char() / put_rx_processed_char(): the actual
//     bridge between our harness's in-memory byte buffers and the modem's
//     tx_process()/rx_process() loops. See harness_io.h.
//   - PERFORM_CPS_TEST / bHighSpeed: fldigi's own existing "run faster
//     than realtime" flags (used for CPS test benches);  we drive these
//     from main.cxx to get non-realtime, full CPU speed TX/RX.
//
// Everything else here is inert: widget pointers stay null, peripheral
// subsystem classes (Cserial, PTT, Caudio_alert, ModeBand, waterfall,
// WFdisp, view_rtty, Fl_ComboBox) get minimal stand-in definitions whose
// only job is to provide a matching mangled symbol - C++ linking only
// cares about the qualified name + parameter types, not which header
// produced the declaration or whether the class "really" has more
// members.

#include "config.h"

#include <string>
#include <cstdio>
#include <cstdint>
#include <complex>

#include "globals.h"     // trx_mode, mode_info[]
#include "soundconf.h"   // SND_IDX_NULL and friends (used by CONFIG_LIST defaults)
#include "configuration.h" // struct configuration, CONFIG_LIST, progdefaults
                           // (transitively pulls in the REAL declarations of
                           // Cserial, PTT, ModeBand, view_rtty, waterfall,
                           // WFdisp, Fl_ComboBox, fre_t, re_t - we provide
                           // out-of-line no-op bodies for their few
                           // referenced-but-undefined methods below, instead
                           // of redeclaring separate stand-in classes.)
#include "status.h"       // struct status, progStatus
#include "digiscope.h"    // Digiscope::scope_mode (real header, class never instantiated)
#include "threads.h"      // SET_THREAD_ID, FLMAIN_TID
#include "qrunner.h"      // cbq[], qrunner::flush()
#include "audio_alert.h"  // Caudio_alert (real header, class never instantiated)
#include "dtmf.h"         // cDTMF (constructed, but never fed real audio)
#include "synop.h"        // synop, SynopDB (weather decode, RTTY rx_init() touches it)

#include <FL/Fl_Button.H>
#include <FL/Fl_Counter.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Input_Choice.H>
#include <FL/Fl_Menu_Item.H>
#include "flslider2.h"    // Fl_Counter2
#include "flinput2.h"     // Fl_Input2

#include "harness_io.h"

// dr_mp3 implementation (normally compiled once in soundcard/play.pa.cxx,
// which we don't link because it's the portaudio alert-sound player).
// SoundBase::AudioMP3() references it but our harness never calls that
// path (it's for the "<AUDIO:file.mp3>" prerecorded-playback macro). The
// header's include guard only allows the implementation to be emitted
// once per translation unit, so this must be the only #include of it.
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

// Config/status globals - real types, real defaults.
//
// configuration.h declares "struct configuration { CONFIG_LIST ... }"
// with ELEM_ bound to ELEM_DECLARE_CONFIGURATION (expands each entry to
// "type var;"). To get the *default value* list instead (what
// configuration.cxx does to define the real progdefaults), redefine
// ELEM_ to only emit the default-value tail of each entry before
// re-expanding CONFIG_LIST here.
#define ELEM_PROGDEFAULTS(type_, var_, tag_, doc_, ...) __VA_ARGS__,
#undef ELEM_
#define ELEM_ ELEM_PROGDEFAULTS
configuration progdefaults = { CONFIG_LIST };

status progStatus = {
	MODE_MFSK128L, "MFSK128L", 0, 1.0, 50, 50, 800, 600,
	false, false, false, false
};

// Threading: qrunner. We run single-threaded and always identify as the
// "main" thread (see main.cxx: SET_THREAD_ID(FLMAIN_TID)), so the
// REQ()/REQ_SYNC() macros throughout the linked modem code never touch
// cbq[] at all - they just call the bound function directly inline.
// cbq[] and qrunner::flush() only need to exist to satisfy the linker.
qrunner *cbq[NUM_QRUNNER_THREADS] = { };
void qrunner::flush(void) { }

// TX/RX text seam - the real bridge to our harness buffers.
int get_tx_char(void)
{
	return harness_get_tx_char();
}

void put_rx_char(unsigned int data, int style)
{
	(void)style;
	harness_put_rx_char(data);
}

void put_rx_processed_char(unsigned int data, int style)
{
	(void)style;
	harness_put_rx_char(data);
}

void put_sec_char(char) { }
int get_secondary_char() { return -1; }
void put_echo_char(unsigned int, int) { }
char *get_rxtx_data() { static char c = 0; return &c; }
char *get_rx_data() { static char c = 0; return &c; }
char *get_tx_data() { static char c = 0; return &c; }
void put_rx_data(int*, int) { }

// Status / metric / scope callbacks - all no-ops, we don't render a UI.
void put_status(const char*, double, status_timeout) { }
void put_Status1(const char*, double, status_timeout) { }
void put_Status2(const char*, double, status_timeout) { }
void put_WARNstatus(double) { }
void put_MODEstatus(long) { }
void put_Bandwidth(int) { }
void put_freq(double) { }
void put_cwRcvWPM(double) { }
void callback_set_metric(double) { }
void set_scope(double*, int, bool) { }
void set_video(double*, int, bool) { }
void set_zdata(std::complex<double>*, int) { }
void set_phase(double, double, bool) { }
void set_scope_mode(Digiscope::scope_mode) { }
void show_bw(std::string) { }
void show_mode(std::string) { }
void show_frequency(unsigned long long) { }
void show_clock(bool) { }
void show_spot(bool) { }
void showDTMF(std::string) { }
void UI_select(std::string) { }
void UI_STATS(std::string) { }
void update_main_title() { }
void updateOutSerNo() { }
void redraw_windows() { }
void restoreFocus(int) { }
void note_qrg(bool, const char*, const char*, long, unsigned long long, int) { }
void qsy(unsigned long long, int) { }
void set599() { }
void setwfrange() { }
void toggle_visible_modes(void*, void*) { }
void toggleRSID() { }

// TX/RX control - our harness drives tx_init/tx_process/rx_process
// directly, so these hooks (normally used by the GUI T/R button and
// macro engine) are no-ops.
void abort_tx() { }
void start_tx() { }
void set_rx_tx() { }
void set_rx_only() { }
void resetRTTY() { }
void resetFSK() { }
void resetCONTESTIA() { }
void resetDOMEX() { }
void resetOLIVIA() { }
void resetTHOR() { }
void resetSoundCard() { }
void reset_mnuPlayback() { }
void Rx_queue_execute() { }
void startMacroTimer() { }
void stopMacroTimer() { }
void start_deadman() { }
void stop_deadman() { }
void rsid_eot_squelch() { }
void init_modem(long, int) { }
void init_modem_sync(long, int) { }
void init_modem_squelch(long, int) { }

// Queues / flags normally owned by the macro engine / GUI event loop.
bool PERFORM_CPS_TEST = false;
bool que_ok = true;
bool tx_queue_done = true;
bool oktoclear = true;
bool bWF_only = false;
bool bEXITING = false;
bool first_use = false;
int  trx_inhibit = 0;
int  Qidle_time = 0;
int  Qwait_time = 0;
bool arqmode = false;
bool tlfio = false;
bool connected_to_flrig = false;
bool CW_KEYLINE_isopen = false;
bool use_nanoIO = false;
bool use_Nav = false;
int  data_io_enabled = 0;

// String globals: directory paths / titles / device names. None of the
// filesystem features (logging, macros, wrap/flmsg folders, waterfall
// palettes...) are exercised headlessly, so empty strings are fine.
std::string HomeDir, DebugDir, AnalysisDir, AvatarDir, DATA_dir;
std::string FLMSG_dir, FLMSG_WRAP_auto_dir, FLMSG_WRAP_recv_dir, FMTDir;
std::string HelpDir, LogsDir, LoTWDir, MacrosDir, PalettesDir, PicsDir;
std::string PskMailDir, ScriptsDir, TalkDir, TempDir, WRAP_auto_dir;
std::string main_window_title, version_text, xcvr_title, build_text;
std::string fsq_selected_call;
std::string scDevice[2];
char szTestChar[] = "E|I|S|T|M|O|A|V";

// Misc free functions referenced from the linked modem/sound/rsid code
// for optional features we leave disabled via progdefaults (RSID
// auto-notify, PSKmail S/N reporting, PSM carrier squelch, CW/FSK
// hardware keying, FLRIG rig CAT, heard-list logging, ...).
void notify_rsid(trx_mode mode, int freq) { fprintf(stderr, "[RSID_CHECK] notify_rsid: mode=%s (%d) freq=%d\n", mode_info[mode].sname, (int)mode, freq); }
void notify_rsid_eot(trx_mode, int) { }
bool bcast_rsid_kiss_frame(int, int, int, int, int) { return false; }
void pskmail_notify_rsid(trx_mode) { }
void pskmail_notify_s2n(double, double, double) { }
void psm_reset_histogram() { }
void psm_transmit() { }
void psm_transmit_ended(int) { }
void cwio_key(int) { }
void cwio_ptt(int) { }
void nanoCW_tune(int) { }
void nano_send_char(int) { }
void Nav_send_char(int) { }
void WKFSK_send_char(int) { }
void WK_tune(bool) { }
void set_flrig_ptt(int) { }
int  flrig_get_baud() { return 0; }
int  flrig_get_idles() { return 0; }
int  flrig_get_stopbits() { return 0; }
void flrig_fskio_send_text(std::string) { }
void WriteARQ(unsigned char) { }
void center_rxfilt_at_track() { }
void debug_dialog() { }
int setup_nls() { return 0; }

void *fft_modem = nullptr;
void *spectrum_viewer = nullptr;
void *test_signal_window = nullptr;
void *dlgViewer = nullptr;

// Peripheral subsystem classes: configuration.h (included above) already
// transitively pulls in the REAL declarations of these (serial.h, ptt.h,
// squelch_status.h, view_rtty.h, waterfall.h, combo.h, re.h) since
// fldigi's own headers are that tangled. We are not linking their .cxx
// implementations (serial port I/O, PTT hardware keying, the RTTY text
// viewer, waterfall display, the custom combo-box widget), so we supply
// out-of-line no-op bodies here for just the handful of methods actually
// referenced by the modem/sound code we do link.
Cserial::Cserial() { }
Cserial::~Cserial() { }
bool Cserial::OpenPort() { return false; }
void Cserial::ClosePort() { }
void Cserial::SetDTR(bool) { }
void Cserial::SetRTS(bool) { }
Cserial rigio;

PTT::PTT(ptt_t) { }
PTT::~PTT() { }
void PTT::set(bool) { }
void PTT::reset(ptt_t) { }
PTT *push2talk = nullptr;

void Caudio_alert::alert(std::string) { }
void Caudio_alert::monitor(double*, int, int, double) { }
Caudio_alert *audio_alert = nullptr;
void reset_audio_alerts() { }

class Rig;
Rig *xcvr = nullptr;

void ModeBand::init() { }
void ModeBand::load_mode_state() { }
void ModeBand::save_mode_state() { }
void ModeBand::band_mode_change() { }
ModeBand modeband;

view_rtty::view_rtty(trx_mode) : modem() { }
view_rtty::~view_rtty() { }
void view_rtty::init() { }
void view_rtty::rx_init() { }
void view_rtty::restart() { }
int  view_rtty::rx_process(const double*, int) { return 0; }
int  view_rtty::tx_process() { return -1; }
void view_rtty::clear() { }
void view_rtty::clearch(int) { }

void waterfall::opmode() { }
int waterfall::Carrier() { return 0; }
void waterfall::Carrier(int) { }
unsigned long long waterfall::rfcarrier() { return 0; }
void waterfall::rfcarrier(unsigned long long) { }
void waterfall::set_XmtRcvBtn(bool) { }
bool waterfall::USB() { return true; }
void waterfall::USB(bool) { }
waterfall *wf = nullptr;

void WFdisp::handle_sig_data() { }
void WFdisp::makeMarker() { }
double WFdisp::powerDensity(double, double) { return 0.0; }
void WFdisp::sig_data(double*, int) { }

void Fl_ComboBox::add(const char*, void*) { }
void Fl_ComboBox::clear() { }
int  Fl_ComboBox::index() { return 0; }
void Fl_ComboBox::index(int) { }
const char* Fl_ComboBox::value() { return ""; }
void Fl_ComboBox::value(std::string) { }
Fl_ComboBox *cboHamlibRig = nullptr;

re_t::re_t(const char*, int) { }
re_t::~re_t() { }
fre_t::fre_t(const char* pattern_, int cflags_) : re_t(pattern_, cflags_) { }
bool fre_t::match(const char*, int) { return false; }

namespace FSEL {
	const char* select(const char*, const char*, const char*, int*) { return nullptr; }
	const char* saveas(const char*, const char*, const char*, int*) { return nullptr; }
}

namespace icons {
	void set_message_icon(const char**) { }
	const char* make_icon_label(const char* text, const char**) { return text; }
	void set_icon_label(Fl_Menu_Item*) { }
	void set_active(Fl_Menu_Item*, bool) { }
	void set_active(Fl_Widget*, bool) { }
}
// Fl_Input2's right-click edit-menu icons (constructed once per instance;
// we only ever construct inpCall/inpFreq below, never shown).
const char *edit_undo_icon[] = { nullptr };
const char *edit_select_all_icon[] = { nullptr };
const char *edit_clear_icon[] = { nullptr };
const char *edit_copy_icon[] = { nullptr };
const char *edit_cut_icon[] = { nullptr };
const char *edit_paste_icon[] = { nullptr };
const char *trash_icon[] = { nullptr };
const char** dialog_warning_48_icon = nullptr;
void *mnu_debug_level = nullptr;

bool SynopDB::Init(const std::string&) { return false; }
// RTTY's rx_init()/rx_process() unconditionally call synop::instance()->...
// (SYNOP weather-report decoding, irrelevant to AMP-2 but not gated out) -
// a real no-op implementation avoids null-vtable-dispatch crashes.
namespace {
struct synop_noop : public synop {
	void init() override { }
	void cleanup() override { }
	void add(char) override { }
	void flush(bool) override { }
	bool enabled(void) const override { return false; }
};
}
synop* synop::instance() { static synop_noop inst; return &inst; }
const synop_callback* synop::ptr_callback = nullptr;

// graydecode/grayencode/parity: real definitions linked from misc/misc.o

int  IMAGE_WIDTH = 3000;
bool mailclient = false;
bool mailserver = false;

void activate_mfsk_image_item(bool) { }
void set_contestia_tab_widgets() { }
void set_dominoex_tab_widgets() { }
void set_olivia_tab_widgets() { }
void set_rtty_tab_widgets() { }
const char* zdate(void) { return ""; }
const unsigned long zmsec(void) { return 0; }
int fmt_auto_record = 0;

// cDTMF: SSB-monitor DTMF tone squelch, unrelated to our modems. trx.cxx
// unconditionally constructs one in trx_start() (which we never call, but
// its inline constructor still needs cDTMF::row/col to link the TU).
void cDTMF::receive(const float*, size_t) { }
void cDTMF::send() { }
int cDTMF::row[] = { 697, 770, 852, 941 };
int cDTMF::col[] = { 1209, 1336, 1477, 1633 };

// Widget pointers referenced from soundcard/soundconf.cxx (audio device
// selection UI), debug.cxx, and a couple of testsigs/config fields. None
// are ever dereferenced: the code paths that would use them are gated by
// progdefaults flags we leave at their defaults (OSS/PortAudio device
// init dialogs, debug window, noise/offset test dialog).
Fl_Button *btnAudioIO = nullptr;
Fl_Button *btnNoiseOn = nullptr;
Fl_Button *btnOffsetOn = nullptr;
Fl_Button *AudioOSS = nullptr;
Fl_Button *AudioPort = nullptr;
Fl_Button *AudioPulse = nullptr;
Fl_Button *btext = nullptr;
Fl_Button *source_code = nullptr;
Fl_Counter *cntRxRateCorr = nullptr;
Fl_Counter *cntTxOffset = nullptr;
Fl_Counter *cntTxRateCorr = nullptr;
Fl_Counter *ctrl_freq_offset = nullptr;
Fl_Counter2 *noiseDB = nullptr;
// Real (never shown/never entered into the FLTK event loop) instances,
// not null: sound.cxx's tag_file() unconditionally dereferences
// inpFreq->value() when writing WAV metadata comments.
Fl_Input2 *inpCall = new Fl_Input2(0, 0, 1, 1);
Fl_Input2 *inpFreq = new Fl_Input2(0, 0, 1, 1);
Fl_Input *inpPulseServer = nullptr;
Fl_Input_Choice *menuOSSDev = nullptr;
Fl_Input_Choice *menuPortInDev = nullptr;
Fl_Input_Choice *menuPortOutDev = nullptr;
Fl_Input_Choice *menuInSampleRate = nullptr;
Fl_Input_Choice *menuOutSampleRate = nullptr;
Fl_Input_Choice *menuAlertsDev = nullptr;
Fl_Input_Choice *menuSampleConverter = nullptr;
