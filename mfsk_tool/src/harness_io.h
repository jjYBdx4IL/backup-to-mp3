// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vector>
#include <cstdint>

// Bridges fldigi's modem TX/RX text seam (get_tx_char/put_rx_char) to a
// plain in-memory byte buffer, so our driver can feed/collect raw bytes
// without any GUI text widgets involved.

void harness_tx_reset(const uint8_t* data, size_t len);
int  harness_get_tx_char(void); // returns GET_TX_CHAR_NODATA/ETX or a byte 0-255
bool harness_tx_done(void);
size_t harness_tx_pos(void); // count of bytes handed out via harness_get_tx_char so far

void harness_rx_reset(void);
void harness_put_rx_char(unsigned int data);
const std::vector<uint8_t>& harness_rx_bytes(void);
