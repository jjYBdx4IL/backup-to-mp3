# backup-to-mp3

How to use MP3 as storage medium.

## Why though?!?

Some devices like the Huawei Watch D2 are dumb devices. In fact it can only store a limit set of audio formats and the user isn't even able to download them as files per se. The only way to get them back is via recording, either via Google's Bumble framework (imitate a bluetooth headset and receive the SBC stream in D2's case) or via air gap (microphone). So it's either an additional SBC compression/decompression step, or an additional DAC->ADC plus lots of noise conversion step. Pick your poison. With amodem+SBC I achieved 32kbps per second of MP3 playback time/duration (320kbps), with amodem+gap I achieved 4kbps per second of MP3 duration. The GGwave solution yielded around 0.14kbps. amodem is very simple and probably bad for larger files. My window dressing put on top of GGwave should make it somewhat more reasonable for large files if using data redundancy outside (ie WinRAR recovery record).

The final solution I came up with is fldigi+flamp. You can download both here:

* https://sourceforge.net/projects/fldigi/files/

Under Linux you can use the PulseAudio null device to record/play back the audio signal to fldig+flamp, under Windows you might have to you a virtual audio cable for that, at least for the recording phase, assuming you won't record straight to some external recorder device, which, of course, should be possible. Beware that if you are using the virtual audio cable under Windows, it is by default set to 'allow exclusive mode'. You can turn that off, but I prefer to just not let recorder/play back app and (ie Audacity) fldigi use the same device for the same purpose (record or playback) at any time. Also, fldigi pretends to allow recording to file I/O only. I didn't have much success with it. It always aborted recording at some point, maybe because I was telling flamp to switch encoding (modem) for the header.

## Why fldigi+flamp?

Flamp provides a block based file transfer protocol including sending and receiving headers with filename etc. information. Flamp connects to fldigi for audio decoding/encoding from/to various transmission coding standards developed for noisy/unreliable transmission mechanisms. Fldigi allows to add a header that tells the receiver what coding will be used, which I deem very useful so I don't have to remember it should I need it a few years out.

## Settings for fldigi

* TxID/RxID ON
* Windows DirectSound/MME/whatever virtual cable for either input or output, not both at the same time. App restart might work in some cases.

## Settings for flamp

### Configure Page

* enter some Callsign, your initials or whatever, not going public anyway
* auto sync fldigi - ON
* change fldigi mode just prior to TX - ON
* everything else OFF
* enable header mode: OFF - we repeat the transmission multiple times instead - a slower header mode could theoretically provide some stability, but it might also be the cause for fldigi's direct audio capture to stop (just as a side note, I don't recommend using that feature).

### Transmit Page

* send to QST
* block size 64
* xmt rpt 1
* hdr rpt 2 (can't hurt)
* base256, 8PSK1200F for SBC/BT download, MFSK128L for air gap/mic and you don't want to rely on Bumble or similar (about 10x slower)
* comp off
* unproto off
* press Add to select the file to store/transmit/encode
* press xmit for a singular transmission or
* go to events page, enable continuous repeat, and click start events. And let it play until you think you have enough reptitions/redundancy in your output stream recorded.

### Receive Page

* nothing much to do here, it's all automatic.
* once you have received all the blocks, click save.

## Settings for Audacity

* Nothing special. Again, don't use input and output of the virtual cable at the same time (or just disable "allow exclusive access" on the virtual cable in Windows' sound settings).
* Might want to shut down everything else running on your computer. Modern computers aren't built for realtime applications (ie audio recording).
* Maybe increase audio buffer sizes?

## Cautionary Warnings

* Don't forget testing whether you can actually recover your data from your final storage device (ie mp3 player). You might want to use RAR format and add a recovery record to your archive. You can also use zfec, but WinRAR (there is a command line version for Linux btw) makes it more convenient by attaching the redundancy data directly to the archive file.
* The MP3 format uses compression that strips out certain parts of audio, which might be in conflict with the modem/encoding in use. That's why I tried amodem's OFDM, which seems particularly well suited for MP3 compression because it's directly using the frequency domain.
* It's also not the worst idea to copy the same file multiple times onto your final storage. None of that storage likely has data checksums/redundancy built in. I guess in many cases the filesystem structure per se has redundancies and checksumming built in nowadays (mostly probably due to ext4 developments and many filesystems keeping a second copy of metadata around), so you usually don't see devices crapping out outright and completely that often. But single apps/files don't get protected from bit rot or whatever might be happening on there. So don't be surprised if your stored mp3 file suddenly turns out to be faulty. On top your device's hw might crap out, so don't make it the only backup solution for whatever you are storing there.
* Don't rely on WinRAR's recovery volume feature too much. It has apparently only one header and if that header gets shot, the recovery volume is unusable. I can't remember a single time RAR's repair feature involving the recovery volume actually worked for me in practice. It's probably better to just set the flamp transmission on repeat and record the output stream a few times. The flamp receiver can then automatically fill in the missing blocks.
* Don't run the recording process (data -> MP3) inside a virtual machine. The clock precision of VMs is infamous.
* Audacity shows recording interruptions/gaps. One more reason to use that piece of software.

## RESULTS

Let's define storage density as bits per second of playback time.

| Method | Storage Density | Notes |
| :--- | :--- | :--- |
| GGwave | 0.14kbps | max encoding length 140 bytes, with my wrapper it might be possible to use it for larger files - unless you start considering the low storage density |
| amodem+SBC | 32kbps | one error, and the stream is gone |
| amodem+gap | 4kbps | one error, and the stream is gone |
| fldigi+flamp 8PSK1200F | ~1kbps | 32kHz 64kbps mono MP3 recoverable via SBC/BT, not air gap |
| fldigi+flamp MFSK128L | ~0.1kbps | OFDM with internal FEC |

FEC: forward error correction

## Other Notes

There are three subdirs:

* amodem-eval/ is my evaluation code around amodem.
* ggwave-build/ is my build/evaluation code around GGWave.
* ofdm/ is my own shot at OFDM in python.

## My Preferred Choice

* WinRAR 7+ recovery record of 100% redundancy plus password protection (`rar a -m5 -rr100 -p recovery.rar your files ...`). Anything beyond 5% is probably waste of space/time. The flamp repetition is much safer IMHO. But since I use that method to have an additional backup copy of my primary keys, whatever...
* fldigi+flamp MFSK128L on repeat into a single mono MP3 with 64kpbs data rate and 32 kHz sampling rate. It's horribly slow, but it decodes well via mic and SBC, and it has automatic error recovery built in (separately!) into MFSK128L and flamp (if you record it multiple times). You also don't need to remember the encoding params or format because the encoded header contains all that information. You just need to remember that you used flamp or HAM radio stuff - a search of the latter should eventually turn up flamp.
* Copy that MP3 multiple times onto the target device to have redundancy against bit rot on that device.
* Perform a test recovery from those files (incl test of the stored files/keys/pwds etc. - the encoded archive/files themselves might have been faulty when accessed during archival or transmission (faulty RAM, random bit flip, millions of potential causes because modern consumer PCs are fast but without error correction/ECC - sometimes even the harddisks suppress checksum errors and just deliver the bad sector data)).
* Be aware that any non-monitored backup dataset can just go bad without any warning. Just a one-off copy to your non-monitored mp3 storage might be even less reliable than you think (apart from bit rot, an automatic device update might clean your device, you might forget you had that backup on it when resetting the device to make it work again without noticing, etc etc etc)

That way I get 6 KB of raw, non-redundant archive data 5 times into one hour of playback (= one 23MB MP3 file).

