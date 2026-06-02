from pathlib import Path
import argparse
from amadeo_utils.media_utils.audio_utils import SimpleAudioManipulation

"""
This simply 'cleans up' audio so the human voice is more prominant - it removes background buzzing, electrical humming, and general noise from the audio file. 

You can either use --file for a single file (specify the file name), or --dir to specify an entire directory.  
 
"""
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Noise Reduction')
    parser.add_argument('file_path', help='Either a specific file or directory to clean.')

    args = parser.parse_args()

    # The attribute name will still be 'file_path'
    if args.file_path and Path(args.file_path).is_file():
        # This is a single file
        print(f"File {args.file_path} will be cleaned.")
        output_file = SimpleAudioManipulation.simple_audio_noise_reduction(args.file_path)
        if output_file:
            print(f"File {args.file_path} cleaned and saved as {output_file}.")
        else:
            print(f"File {args.file_path} failed to be cleaned.")

    elif args.file_path and Path(args.file_path).exists():  # Fixed this line
        # This is a directory
        filesCleaned = SimpleAudioManipulation.noise_reduction_dir(args.file_path, args.file_path)
        if filesCleaned:
            str_files = ",".join(str(file) for file in filesCleaned)
            print(f"Files that were cleaned in {args.file_path}: {str_files}.")
        else:
            print(f"No files in {args.file_path} to be cleaned.")
    else:
        print(f"You did not provide a valid file nor directory to clean.")



