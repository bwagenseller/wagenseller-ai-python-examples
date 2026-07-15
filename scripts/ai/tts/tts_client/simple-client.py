"""
TTS Client - Sends JSON requests to an F5-TTS server and plays received audio files
"""

import json
import struct
import time
from typing import Dict
from pathlib import Path
import logging
from amadeo_utils.client.amadeo_client import AmadeoClient

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, host='localhost', port=8888, output_dir='tts_output'):
        self.host = host
        self.port = port

        self.socket_client = AmadeoClient(self.host, self.port, additional_server_response_functionality = self.handle_server_response)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"TTS Client - Connected to {self.host}:{self.port}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        # Try to import audio playback library
        self.audio_player = None
        try:
            import pygame
            pygame.mixer.init()
            self.audio_player = 'pygame'
            logger.info("Using pygame for audio playback")
        except ImportError:
            try:
                import playsound
                self.audio_player = 'playsound'
                logger.info("Using playsound for audio playback")
            except ImportError:
                logger.warning("No audio playback library found. Install pygame or playsound for audio playback.")
    
    def generate_filename(self, voice, text):
        """Generate a unique filename for the audio file"""
        timestamp = int(time.time())
        # Clean text for filename
        clean_text = ''.join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_text = clean_text.replace(' ', '_')
        
        base_filename = f"{voice}_{clean_text}_{timestamp}.wav"
        return self.output_dir / base_filename
    
    def play_audio(self, file_path):
        """Play audio file using available library"""
        if not self.audio_player:
            logger.warning("No audio player available")
            return
        
        try:
            if self.audio_player == 'pygame':
                import pygame
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play()
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
            elif self.audio_player == 'playsound':
                from playsound import playsound
                playsound(str(file_path))
            
            logger.info(f"Played audio file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")


    def handle_server_response(self, response:Dict, raw_data:bytes = None) -> None:

        file_size = 0
        if raw_data: file_size = len(raw_data)

        if response['success'] == True:

            if response['type'] == 'audio':
                logger.info(f"Receiving audio file ({file_size} bytes)")

                # Save audio file with unique name
                output_file = self.filename
                with open(output_file, 'wb') as f:
                    f.write(raw_data)

                # Play audio
                self.play_audio(output_file)

                logger.info(f"Saved audio file to: {output_file}")
            elif response['type'] == 'voices':
                local_voices = response.get('message', 'Unknown')
                logger.info(f"Here are the available voices: {local_voices}")
            else:
                logger.warning(f"The call to the server succeeded, but we dont know WHAT succeeded. Message: {response['message']}")
        else:
            error_msg = response.get('message', 'Unknown error')
            logger.error(f"Server error: {error_msg}")

    def set_filename(self, voice, text):
        self.filename = self.generate_filename(voice, text)

    def interactive_mode(self):
        """Run interactive mode for testing"""
        print("\nEnter '!quit' to exit\n")
        print("Enter '!voices' to see voices\n")
        
        while True:
            try:
                text = input("Enter either !quit, !voices, or text to convert to speech: ").strip()
                if text.lower() == '!quit':
                    break
                if text.lower() == '!voices':
                    self.socket_client.send_transient_request('show_voices')

                else:
                    if not text:
                        continue

                    voice = input("Enter voice name (or press Enter for 'default'): ").strip()
                    if not voice:
                        voice = 'default'

                    logger.info("Sending request...")

                    self.set_filename(voice, text)
                    self.socket_client.send_transient_request('service_tts', '', text=text, voice=voice)

            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
            except Exception as e:
                logger.info(f"Error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TTS Client')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--output', default='tts_output', help='Output directory for audio files')
    parser.add_argument('--text', help='Text to convert to speech')
    parser.add_argument('--voice', default='default', help='Voice to use')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    client = TTSClient(args.host, args.port, args.output)
    
    if args.interactive:
        client.interactive_mode()
    elif args.text:
        client.set_filename(args.voice, args.text)
        client.socket_client.send_transient_request('service_tts', '', text=args.text, voice=args.voice)
    else:
        parser.print_help()
    
    return 0

if __name__ == '__main__':
    exit(main())