// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone diagnostic (not part of the shipped tool): reads the raw
// decoded header/payload text dumped via AMP_DUMP_DIR and reports, per
// data-block repeat, how many blocks were recoverable - to answer
// REQUIREMENTS.md's "does it need repeats, or does the first pass work"
// question empirically instead of by inspection.
#include "amp_proto.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

static std::string slurp(const char* path)
{
	std::ifstream f(path, std::ios::binary);
	std::ostringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

int main(int argc, char** argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s <hdr_text.txt> <mfsk_text.txt>\n", argv[0]); return 2; }
	std::string hdr_text = slurp(argv[1]);
	std::string mfsk_text = slurp(argv[2]);

	// Split payload text into repeat segments on "{hash:EOF}" boundaries.
	std::vector<std::string> segments;
	size_t start = 0;
	while (true) {
		size_t eof = mfsk_text.find(":EOF}", start);
		if (eof == std::string::npos) { segments.push_back(mfsk_text.substr(start)); break; }
		size_t end = mfsk_text.find('\n', eof);
		if (end == std::string::npos) end = mfsk_text.size(); else end += 1;
		segments.push_back(mfsk_text.substr(start, end - start));
		start = end;
	}
	fprintf(stderr, "split into %zu segment(s)\n", segments.size());

	// Per-repeat alone (each combined with the header text, needed for SIZE/numblocks).
	for (size_t i = 0; i < segments.size(); i++) {
		AmpReceiveState st;
		amp_parse_lines(hdr_text, st);
		amp_parse_lines(segments[i], st);
		fprintf(stderr, "repeat %zu alone: blocks=%zu/%d complete=%s missing=",
			i + 1, st.blocks.size(), st.numblocks, amp_rx_complete(st) ? "YES" : "no");
		for (int b = 1; b <= st.numblocks; b++)
			if (st.blocks.find(b) == st.blocks.end()) fprintf(stderr, "%d ", b);
		fprintf(stderr, "\n");
	}

	// Cumulative (repeat 1, repeat 1+2, repeat 1+2+3, ...).
	AmpReceiveState cum;
	amp_parse_lines(hdr_text, cum);
	for (size_t i = 0; i < segments.size(); i++) {
		amp_parse_lines(segments[i], cum);
		fprintf(stderr, "cumulative through repeat %zu: blocks=%zu/%d complete=%s\n",
			i + 1, cum.blocks.size(), cum.numblocks, amp_rx_complete(cum) ? "YES" : "no");
	}
	return 0;
}
