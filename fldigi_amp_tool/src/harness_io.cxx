// SPDX-License-Identifier: GPL-3.0-or-later
#include "harness_io.h"

#define GET_TX_CHAR_NODATA -1
#define GET_TX_CHAR_ETX -3

namespace {
	std::vector<uint8_t> tx_buf;
	size_t tx_pos = 0;
	bool tx_etx_sent = false;

	std::vector<uint8_t> rx_buf;
}

void harness_tx_reset(const uint8_t* data, size_t len)
{
	tx_buf.assign(data, data + len);
	tx_pos = 0;
	tx_etx_sent = false;
}

int harness_get_tx_char(void)
{
	if (tx_pos < tx_buf.size())
		return tx_buf[tx_pos++];
	tx_etx_sent = true;
	return GET_TX_CHAR_ETX;
}

bool harness_tx_done(void)
{
	return tx_etx_sent;
}

size_t harness_tx_pos(void)
{
	return tx_pos;
}

void harness_rx_reset(void)
{
	rx_buf.clear();
}

void harness_put_rx_char(unsigned int data)
{
	rx_buf.push_back(static_cast<uint8_t>(data & 0xFF));
}

const std::vector<uint8_t>& harness_rx_bytes(void)
{
	return rx_buf;
}
