import argparse
from pathlib import Path
from amadeo_utils.media_utils.audio_utils import SimpleAudioManipulation

"""
This is the main file to extract audio
"""

def main():
    parser = argparse.ArgumentParser(
        description="Extract multiple audio segments and join with silence",
        epilog='''
Examples:
  python script.py input.mp4 output.wav "(5,15), (25,10), (40,35)"
  python script.py video.webm result.wav "(0,30), (60,45)" --silence 1.0
  python script.py audio.mp3 out.wav "(10,5)" --sample-rate 22050
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_file", help="Input audio/video file (.mp4, .m4v, .wmv, .webm, .mov, .mp3, .wav)")
    parser.add_argument("output_file",help="Output WAV file")
    parser.add_argument("--time_segments",default="",help='Time segments as tuples: "(start1,duration1), (start2,duration2), ..." in seconds. Can be float values.')
    parser.add_argument("--silence", "-s",type=float,default=0.5,help="Seconds of silence between segments (default: 0.5)")
    parser.add_argument("--sample-rate", "-sr",type=int,default=24000,help="Target sample rate (default: 24000)")


    args = parser.parse_args()
    argDict = {}


    argDict['input_file'] = args.input_file
    argDict['output_file'] = args.output_file
    argDict['time_segments'] = args.time_segments
    argDict['silence'] = args.silence
    argDict['sample-rate'] = args.sample_rate


    # Validate input file exists
    input_path = Path(argDict['input_file'])
    if not input_path.exists():
        print(f"❌ Input file not found: {argDict['input_file']}")
        return 1

    segments = SimpleAudioManipulation.get_sample_rate(argDict['input_file'])
    print(f"Sample rate of the original file: {segments}")

    # Validate output file extension
    output_path = Path(argDict['output_file'])
    if output_path.suffix.lower() != '.wav':
        print(f"❌ Output file must be .wav, got: {output_path.suffix}")
        return 1

    # Process the audio
    manipulator = SimpleAudioManipulation()

    if argDict['time_segments'] != "":
        # if the time segments were present, this means we are parsing one or multiple parts. Parse time segments
        try:
            segments = SimpleAudioManipulation.parse_time_segments(argDict['time_segments'])
            print(f"🎯 Parsed {len(segments)} time segments:")
            for i, (start, duration) in enumerate(segments):
                print(f"   Segment {i+1}: {start}s → {start+duration}s ({duration}s)")
        except ValueError as e:
            print(f"❌ {e}")
            print("Format: \"(start1,duration1), (start2,duration2), ...\"")
            print("Example: \"(5,15), (25,10), (40,35)\"")
            return 1

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            combined_audio, sample_rate = manipulator.extract_multiple_samples(
                argDict['input_file'],
                segments,
                argDict['sample-rate'],
                argDict['silence']
            )

            if combined_audio is None:
                print("❌ Failed to extract segments")
                return 1

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return 1
    else:
        # if the time segments were not present, do the entire file

        try:
            combined_audio, sample_rate = manipulator.extract_full_audio(
                argDict['input_file'],
                argDict['sample-rate']
            )

            if combined_audio is None:
                print("❌ Failed to extract segments")
                return 1

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return 1


    # Save the result
    result_file = manipulator.write_audio_data_to_file(
        argDict['output_file'],
        combined_audio,
        sample_rate
    )

    print(f"\n🎉 Success!")
    print(f"📁 Saved: {result_file}")
    print(f"📊 File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return 0

if __name__ == "__main__":
    main()