// SPDX-License-Identifier: GPL-3.0-or-later
#include "amp_proto.h"
#include <cstdio>
#include <cstring>
#include <ctime>

// CRC16 - verbatim port of flamp's Ccrc16 (flamp-2.2.14/src/include/crc16.h)
uint16_t amp_crc16(const std::string& s)
{
	uint32_t crc = 0xFFFF;
	for (unsigned char c : s) {
		crc ^= c;
		for (int i = 0; i < 8; i++)
			crc = (crc & 1) ? (0xA001 ^ (crc >> 1)) : (crc >> 1);
	}
	return (uint16_t)(crc & 0xFFFF);
}

static std::string crc16_hex(const std::string& s)
{
	char buf[8];
	snprintf(buf, sizeof(buf), "%04X", amp_crc16(s));
	return std::string(buf);
}

// base64 - verbatim port of flamp's base64 class with crlf=false
// (flamp-2.2.14/src/utils/base64.cxx): standard alphabet, '=' padding,
// no line wrapping.
static const char* B64ET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string amp_base64_encode(const std::string& in)
{
	std::string out;
	size_t i = 0, n = in.size();
	while (i < n) {
		unsigned char b0 = in[i], b1 = (i+1<n)?in[i+1]:0, b2 = (i+2<n)?in[i+2]:0;
		int rem = (int)(n - i);
		out += B64ET[b0 >> 2];
		out += B64ET[((b0 & 3) << 4) | (b1 >> 4)];
		out += (rem > 1) ? B64ET[((b1 & 0xF) << 2) | (b2 >> 6)] : '=';
		out += (rem > 2) ? B64ET[b2 & 0x3F] : '=';
		i += 3;
	}
	return out;
}

static int b64val(unsigned char c)
{
	if (c >= 'A' && c <= 'Z') return c - 'A';
	if (c >= 'a' && c <= 'z') return c - 'a' + 26;
	if (c >= '0' && c <= '9') return c - '0' + 52;
	if (c == '+') return 62;
	if (c == '/') return 63;
	return -1;
}

std::string amp_base64_decode(const std::string& in)
{
	std::string out;
	int vals[4]; int n = 0;
	for (unsigned char c : in) {
		if (c == '=' ) break;
		int v = b64val(c);
		if (v < 0) continue; // whitespace / noise tolerant, matches flamp's "while (c <= ' ')" skip
		vals[n++] = v;
		if (n == 4) {
			out += (char)((vals[0] << 2) | (vals[1] >> 4));
			out += (char)(((vals[1] & 0xF) << 4) | (vals[2] >> 2));
			out += (char)(((vals[2] & 0x3) << 6) | vals[3]);
			n = 0;
		}
	}
	if (n == 2) {
		out += (char)((vals[0] << 2) | (vals[1] >> 4));
	} else if (n == 3) {
		out += (char)((vals[0] << 2) | (vals[1] >> 4));
		out += (char)(((vals[1] & 0xF) << 4) | (vals[2] >> 2));
	}
	return out;
}

// AMP-2 message assembly (amp.cxx: file_header/id_header/size_header/
// program_header/data_block/data_eof/data_eot/xmt_vector_string)
static std::string amp_line(const char* ltype, const std::string& content)
{
	char lenbuf[16];
	snprintf(lenbuf, sizeof(lenbuf), "%d", (int)content.size());
	std::string out = ltype;
	out += lenbuf;
	out += " ";
	out += crc16_hex(content);
	out += ">";
	out += content;
	out += "\n";
	return out;
}

// MFSK128L interleaver warm-up/cooldown padding. Real FLAMP transfers are
// large enough that the payload itself fills interleave.cxx's depth=800
// shift-register cascade before/after the file's actual bytes; a payload as
// tiny as a handful of AMP-2 blocks never does, leaving its edge bits with
// less effective FEC spread than the middle of a real transfer gets - the
// same effect docs/bitrate_sweep_report.txt bugs #2/#3 hit and fixed (there,
// in bitrate_sweep.cxx's build_warmup()) for MFSK128L runs with no head/tail
// margin. These filler lines contain none of amp_parse_lines()'s "<TYPE "
// tags, so any receiver's line parser (real FLAMP's included - its own
// parser is the same skip-unrecognized-bytes design ours here is a verbatim
// port of) just treats them as noise; they exist purely to keep real signal
// flowing through the modem's encode/decode pipeline around the genuine
// data, not to carry information.
static const int PAD_LINES = 20;
static const int PAD_LINE_LEN = 64;

static std::string amp_padding()
{
	std::string out;
	for (int i = 0; i < PAD_LINES; i++) {
		out += std::string(PAD_LINE_LEN, '0' + (i % 10));
		out += "\n";
	}
	return out;
}

static std::string utc_dttm()
{
	time_t now = time(nullptr);
	struct tm t;
	gmtime_r(&now, &t);
	char buf[32];
	strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &t);
	return std::string(buf);
}

AmpMessage amp_build_message(const std::vector<uint8_t>& raw_file, const AmpFileParams& p)
{
	std::string dttm = utc_dttm();
	std::string raw((const char*)raw_file.data(), raw_file.size());

	// xmt_data(): "[b64:start]" + base64(raw, no wrap) + "\n[b64:end]"
	std::string xmtdata = "[b64:start]";
	xmtdata += amp_base64_encode(raw);
	xmtdata += "\n[b64:end]";

	int numblocks = (int)(xmtdata.size() / p.blocksize + (xmtdata.size() % p.blocksize ? 1 : 0));

	// _file_hash(): CRC16(dttm ":" filename "0"|"1" "base64" blocksize)
	std::string hash_input = dttm + ":" + p.filename;
	hash_input += "0"; // no compression (matches our requirement: 7z is already compressed)
	hash_input += "base64";
	char bsbuf[16]; snprintf(bsbuf, sizeof(bsbuf), "%d", p.blocksize);
	hash_input += bsbuf;
	std::string hash = crc16_hex(hash_input);

	AmpMessage msg;
	msg.hash = hash;

	std::string call_from_to = p.call_to + " DE " + p.my_call + "\n\n";

	char szbuf[64];
	snprintf(szbuf, sizeof(szbuf), "%d %d %d", (int)xmtdata.size(), numblocks, p.blocksize);

	msg.transmissions.resize(p.num_transmissions);
	for (int t = 0; t < p.num_transmissions; t++) {
		AmpTransmission& tx = msg.transmissions[t];

		// --- header (announce) modem content: repeat_header copies, this transmission only ---
		tx.header_text = call_from_to;
		for (int i = 0; i < p.repeat_header; i++) {
			tx.header_text += amp_line("<PROG ", "{" + hash + "}amp2mp3 0.1");
			tx.header_text += amp_line("<FILE ", "{" + hash + "}" + dttm + ":" + p.filename);
			tx.header_text += amp_line("<ID ", "{" + hash + "}" + p.my_call + " ");
			tx.header_text += amp_line("<SIZE ", "{" + hash + "}" + szbuf);
		}

		// --- data (payload) modem content: one copy of every block, this transmission only ---
		tx.data_text += amp_padding(); // warm-up: prime the MFSK128L interleaver before real data
		for (int b = 1; b <= numblocks; b++) {
			char tag[16]; snprintf(tag, sizeof(tag), "{%s:%d}", hash.c_str(), b);
			std::string content = tag;
			content += xmtdata.substr((size_t)(b - 1) * p.blocksize, p.blocksize);
			tx.data_text += amp_line("<DATA ", content);
		}
		tx.data_text += amp_line("<CNTL ", "{" + hash + ":EOF}");
		if (t == p.num_transmissions - 1)
			tx.data_text += amp_line("<CNTL ", "{" + hash + ":EOT}");
		tx.data_text += amp_padding(); // cooldown: flush the interleaver after real data
	}

	return msg;
}

// Receive side (amp.cxx: rx_parse_buffer/rx_add_data/rx_parse_size/...)
static const char* LTYPES[] = { "<FILE ", "<ID ", "<DTTM ", "<SIZE ", "<DESC ", "<DATA ", "<PROG ", "<CNTL " };
enum { _FILE=0, _ID, _DTTM, _SIZE, _DESC, _DATA, _PROG, _CNTL };

void amp_parse_lines(const std::string& text, AmpReceiveState& st)
{
	size_t pos = 0;
	while (pos < text.size()) {
		size_t best_p = std::string::npos;
		int best_type = -1;
		for (int t = _FILE; t <= _CNTL; t++) {
			size_t p = text.find(LTYPES[t], pos);
			if (p != std::string::npos && (best_p == std::string::npos || p < best_p)) {
				best_p = p; best_type = t;
			}
		}
		if (best_p == std::string::npos) break;

		size_t after_prefix = best_p + strlen(LTYPES[best_type]);
		int len; char crc[8];
		if (sscanf(text.c_str() + after_prefix, "%d %4s", &len, crc) != 2 || len < 0 || len > 4096) {
			pos = best_p + 1;
			continue;
		}
		size_t gt = text.find('>', after_prefix);
		if (gt == std::string::npos) { pos = best_p + 1; continue; }
		if (gt + 1 + (size_t)len > text.size()) { pos = best_p + 1; continue; }

		std::string content = text.substr(gt + 1, len);
		bool crc_ok = crc16_hex(content) == std::string(crc);
		// Matches real FLAMP's rx_parse_buffer() (amp.cxx:1133-1142): only
		// skip past the full claimed length on a *successful* parse. On a
		// CRC mismatch the claimed length may itself be corrupted noise, so
		// back off by a single byte instead - jumping by a bogus length can
		// leap straight over a later, genuinely valid copy of the same line
		// (observed: a corrupted repeat's garbled "<DATA ...>" claiming a
		// huge length skipped past the next repeat's clean, CRC-valid copy
		// of the same block).
		if (!crc_ok) { pos = best_p + 1; continue; }
		pos = gt + 1 + len;

		switch (best_type) {
			case _FILE: {
				if (content.size() < 6 || content[0] != '{') break;
				std::string h = content.substr(1, 4);
				size_t rb = content.find('}');
				if (rb == std::string::npos) break;
				std::string rest = content.substr(rb + 1);
				size_t colon = rest.find(':');
				if (colon == std::string::npos) break;
				st.hash = h;
				st.dttm = rest.substr(0, colon);
				st.filename = rest.substr(colon + 1);
				st.have_file = true;
				break;
			}
			case _SIZE: {
				char h[8]; int fs, nb, bs;
				if (sscanf(content.c_str(), "{%4s}%d %d %d", h, &fs, &nb, &bs) == 4) {
					if (!st.have_file || st.hash == h) {
						st.filesize = fs; st.numblocks = nb; st.blocksize = bs;
						st.have_size = true;
					}
				}
				break;
			}
			case _PROG: {
				size_t rb = content.find('}');
				if (rb != std::string::npos) st.prog = content.substr(rb + 1);
				break;
			}
			case _ID: {
				size_t rb = content.find('}');
				if (rb != std::string::npos) st.call_info = content.substr(rb + 1);
				break;
			}
			case _DATA: {
				if (content.size() < 6 || content[0] != '{') break;
				std::string h = content.substr(1, 4);
				if (st.have_file && h != st.hash) break; // not our transfer
				size_t colon = content.find(':');
				size_t rb = content.find('}');
				if (colon == std::string::npos || rb == std::string::npos || rb < colon) break;
				int blknbr = atoi(content.substr(colon + 1, rb - colon - 1).c_str());
				if (blknbr > 0 && st.blocks.find(blknbr) == st.blocks.end())
					st.blocks[blknbr] = content.substr(rb + 1);
				if (!st.have_file) st.hash = h; // tolerate data arriving before/without a clean FILE line
				break;
			}
			default: break;
		}
	}
}

bool amp_rx_complete(const AmpReceiveState& st)
{
	if (!st.have_size || st.numblocks <= 0) return false;
	for (int b = 1; b <= st.numblocks; b++)
		if (st.blocks.find(b) == st.blocks.end()) return false;
	return true;
}

std::vector<uint8_t> amp_rx_extract_file(const AmpReceiveState& st)
{
	std::string xmtdata;
	for (int b = 1; b <= st.numblocks; b++) {
		auto it = st.blocks.find(b);
		if (it != st.blocks.end()) xmtdata += it->second;
	}
	size_t s = xmtdata.find("[b64:start]");
	size_t e = xmtdata.find("[b64:end]");
	std::string b64 = (s != std::string::npos && e != std::string::npos && e > s)
		? xmtdata.substr(s + 11, e - (s + 11))
		: xmtdata;
	std::string raw = amp_base64_decode(b64);
	return std::vector<uint8_t>(raw.begin(), raw.end());
}
