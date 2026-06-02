"""
TTS Server - Receives JSON requests with voice and text, returns WAV files using f5-tts

This server listens for client connections, receives JSON requests containing text to speak
and a voice identifier, uses F5-TTS to generate speech audio, and sends the resulting
WAV file back to the client.

Protocol:
1. Client connects via TCP socket
2. Client sends 4-byte header with JSON length
3. Client sends JSON: {"text": "...", "voice": "..."}
4. Server processes with F5-TTS
5. Server sends response header + JSON status
6. Server sends WAV file data (if successful)
7. Connection closes

Key Features:
- Model loads once and stays in memory for fast successive calls
- Graceful shutdown with Ctrl-C (no hanging)
- Proper socket cleanup and timeout handling
- Supports multiple voice configurations via voices.json


Examples:
  python f5-simple-server.py                                    # Start on localhost:8888
  python f5-simple-server.py --host 0.0.0.0 --port 9999      # Listen on all interfaces, port 9999
  python f5-simple-server.py --voices /path/to/my/voices      # Use custom voice directory
  python f5-simple-server.py --json /path/to/json/config/file/f5-tts-simple-server.json
"""

import warnings

"""
Set warning suppression - the important warnings you should know about these:

In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder 
instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.

torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into 
TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release.  

jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
"""
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from pathlib import Path
from amadeo_utils.ai.tts import f5
from amadeo_utils.server.amadeo_server import AmadeoServer
import logging
from typing import Dict

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Set logging levels
logging.getLogger('f5_tts').setLevel(logging.WARNING)

class TTSServer:

    def __init__(self, config_dict:Dict):

        self.f5_model = f5.AmadeoF5(config_dict)
        self.server = AmadeoServer(config_dict['host'], config_dict['port'], additional_client_functionality = self.f5_model.handle_client_request)


def main():

    arg_dict = f5.AmadeoF5.get_args_dict()

    # Create and start the TTS server
    try:
        logger.info("Starting F5-TTS Server...")
        server = TTSServer(arg_dict)
        server.server.start_server()
    except KeyboardInterrupt:
        # This shouldn't happen anymore with proper signal handling, but just in case
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        return 1

    return 0

# Standard Python idiom - only run main() if this script is executed directly
# (not if it's imported as a module)
if __name__ == '__main__':
    exit(main())