

# RESULTS

## Huawei Watch D2

With up to 4 kbps (bits per playback second, ie one second of MP3 playback on the watch contains ~500 bytes) I manage to download a 12 KB file via microphone.

With up to 32 kbps I manage to download the same file via Bumble/bluetooth headset SBC codec.

Shorter files seem to manage higher rates. I included a patch to ignore checksum errors. However, it seems that the protocol isn't suitable for larger transfers. It seems to me that there is no error recovery once a significant part of the protocol gets damaged, so while we could recover from checksum errors by appending redundancy data ourselves (ie out of band), it's only a matter of time those data errors affect the protocol data itself and completely destroy the stream. The next step could be to use amodem to just encode on the block level and limit block sizes and include redundancy data for recovery outside of amodem, ie. make sure we can re-assemble the data stream/file block by block even if the of the blocks are not recoverable.
