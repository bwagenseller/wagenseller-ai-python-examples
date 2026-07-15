"""
Audio device selection helpers for sounddevice / PortAudio.

Kept separate from audio_utils.py so that callers which only need device routing
do not pull in librosa/noisereduce, and separate from audio_capture.py, which
bypasses PortAudio entirely by spawning parec.
"""

import logging

import sounddevice as sd

logger = logging.getLogger(__name__)


def prefer_pulse_defaults(strict: bool = False) -> bool:
    """
    Point sounddevice's default input/output devices at PulseAudio's current defaults.

    PortAudio exposes one or more "host APIs". On a Linux desktop it normally
    selects the ALSA host API, whose devices are raw `hw:` nodes. Those have two
    properties that break the scripts in this repo:

      1. No sample-rate conversion. Opening a raw device at a rate the hardware
         does not natively support fails with
         `Invalid sample rate [PaErrorCode -9997]` - e.g. asking a Realtek ALC1220
         (44.1/48 kHz only) for the 24 kHz that F5-TTS wants, or the 16 kHz the
         VAD/ASR clients want.
      2. Unstable indices. They renumber whenever hardware is plugged or
         unplugged, so any hard-coded device index silently points at the wrong
         card later.

    Raw ALSA also cannot see Bluetooth sinks at all, and may hide a USB device's
    capture side when PortAudio probes it at a rate that device does not accept.

    The PulseAudio host API (present when PortAudio was built with PulseAudio
    support) avoids all of this: its default devices track whatever the desktop's
    sound settings currently point at, it resamples transparently, and it reaches
    Bluetooth and USB devices normally. This function selects that host API's own
    default devices, so routing follows the desktop rather than a fixed index.

    On a headless host there is typically no PulseAudio/PipeWire session and
    PortAudio offers no PulseAudio host API. There is nothing sensible to switch
    to, so the existing defaults are left untouched and False is returned.

    :param strict: If True, raise RuntimeError when no PulseAudio host API is
                   available rather than silently leaving the defaults alone.
                   Useful on a desktop, where its absence indicates a broken
                   install rather than an expected headless environment.
    :return: True if the defaults were changed, False if they were left as-is.
    :raises RuntimeError: If strict is True and no PulseAudio host API is present.
    """
    # Host API names vary ('PulseAudio', 'pulse'), so match on a lowercase substring.
    index = next(
        (i for i, api in enumerate(sd.query_hostapis()) if 'pulse' in api['name'].lower()),
        None
    )

    if index is None:
        message = ("No PulseAudio host API available in PortAudio; leaving sounddevice "
                   "defaults alone (expected on a headless host).")
        if strict:
            raise RuntimeError(message)
        logger.debug(message)
        return False

    host_api = sd.query_hostapis(index)
    input_device = host_api['default_input_device']
    output_device = host_api['default_output_device']

    # PortAudio reports -1 when a host API has no default device of that direction.
    # Keep the existing default in that case rather than clobbering it with -1.
    current_input, current_output = sd.default.device

    if input_device < 0 and output_device < 0:
        logger.debug("PulseAudio host API exposes no default devices; leaving defaults alone.")
        return False

    sd.default.device = (
        input_device if input_device >= 0 else current_input,
        output_device if output_device >= 0 else current_output,
    )

    logger.debug("sounddevice defaults set to PulseAudio devices: input=%s output=%s",
                 input_device, output_device)

    return True
