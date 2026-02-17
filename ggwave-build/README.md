# ggwave-build

Evaluation of: https://github.com/ggerganov/ggwave

Added ggwave-large-file.py to be able to handle 'larger' content.

Performance: ~18 raw input bytes per second of encoded audio. The major issue is when you have
to play back the MP3 to get the data. When I encode my primary recovery archive with 100% redundancy
data (using WinRAR), I get like 13 KB of data. That's about 12 minutes of playback time, just to give
you an impression of what's possible.

# TODO

GGWave output seems unnecessarily low in raw data density. A frequency domain encoding of the data should achieve maybe 50x more
data density (per MP3 playback time) because of the encoding being natural to the MP3 encoder algorithm,
so ~1KB per second of MP3 playback should be achievable.

# Hints for Huawei Lite-type watches (ie Huawei Watch D2)

Those types of devices don't allow for any debugging connections and file transfers.
But you can encode a limited amount of data in MP3s and upload those. However, you
can't directly download those MP3s from the watch, so you need to record the playback
from the watch somehow.

## Use a microphone

* Able to record to WAV without introducing additional compression artifacts.
* Introduces additional DAC-ADC conversion and environment noise.
* Usually just requires a recording app on your phone.

## Pretend to be a headset

* Might be more convenient but additional bluetooth audio compression threatening data integrity.
* Might not work with every bluetooth dongle.
* On Windows use Zadig (https://zadig.akeo.ie/) to replace the USB driver with the WinUSB driver (top one in selection).

Then use Google's Bumble (Python code available on github, https://github.com/google/bumble, I used v0.0.225) with the following command:

```
py apps/speaker/speaker.py --device-config examples/speaker.json --output record.sbc usb:0 --codec sbc
```

The bluetooth auth data is stored somewhere under %USERPROFILE%/AppData/Local/Google/Bumble. FFmpeg has
no issues converting that data dump if you give it the correct hint with the file extension.

