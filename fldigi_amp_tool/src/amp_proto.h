// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>

// Real AMP-2 protocol framing, as implemented by FLAMP 2.2.14
// (flamp-2.2.14/src/utils/amp.cxx, src/include/amp.h). This is NOT a
// custom format: a genuine, unmodified FLAMP (paired with fldigi) must
// be able to receive and decode what we produce here. Every constant,
// field order, and default below was taken directly from that source
// (no compression, base64 encoding, blocksize 64, repeat_header 1,
// callsign "N0CALL", "QST" to-call - all real FLAMP defaults).

uint16_t amp_crc16(const std::string& s); // Ccrc16 in flamp's crc16.h, verbatim

std::string amp_base64_encode(const std::string& raw);   // flamp's base64 class, crlf=false
std::string amp_base64_decode(const std::string& b64);

struct AmpFileParams {
	std::string my_call = "N0CALL";
	std::string call_to = "QST";
	std::string filename = "archive.7z";
	int blocksize = 64;         // FLAMP default xmtblocksize
	int repeat_header = 2;      // header lines repeated this many times, *within* each transmission
	int num_transmissions = 3;  // requirement: 3 full (header + data) transmissions, for redundancy
};

// One full AMP transmission: a header (announce) segment followed by a
// file-data segment - each is its own logical retransmission, so a
// receiver that missed an earlier transmission entirely still gets a
// fresh header to sync to.
struct AmpTransmission {
	std::string header_text; // send via the header (announce) modem
	std::string data_text;   // send via the payload/data modem
};

struct AmpMessage {
	std::vector<AmpTransmission> transmissions;
	std::string hash; // 4 hex chars, for reference/debugging
};

// Builds the exact byte-for-byte content FLAMP would transmit for a
// single-file, uncompressed, base64, header-modem-enabled transfer.
AmpMessage amp_build_message(const std::vector<uint8_t>& raw_file, const AmpFileParams& params);

// Receive side - mirrors amp.cxx's rx_parse_buffer()/rx_add_data(): scan
// arbitrary (possibly garbled, possibly containing text from either
// modem's decode of the other modem's audio) text for well-formed
// "<TYPE LEN CRC>content" lines, verify each line's own CRC16, then
// reassemble DATA blocks by index and extract the file once EOF/EOT and
// all blocks for the matching hash are present.
struct AmpReceiveState {
	std::string hash;              // transfer id, from the FILE header line
	std::string dttm, filename, call_info, prog;
	int filesize = 0, numblocks = 0, blocksize = 0;
	bool have_file = false, have_size = false;
	std::map<int, std::string> blocks; // 1-based block# -> raw block content (after "{hash:n}")
};

// Feed arbitrary decoded text (from EITHER modem's RX - concatenate
// everything you have, order doesn't matter beyond it being append-only
// per source) into the parser; it finds every well-formed, CRC-valid
// line anywhere in it.
void amp_parse_lines(const std::string& text, AmpReceiveState& st);

// True once every block 1..numblocks has been collected for the
// established hash.
bool amp_rx_complete(const AmpReceiveState& st);

// Reassembles blocks in order, strips the [b64:start]/[b64:end] markers,
// base64-decodes, and returns the reconstructed original file bytes.
std::vector<uint8_t> amp_rx_extract_file(const AmpReceiveState& st);
