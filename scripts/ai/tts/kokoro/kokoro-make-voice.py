import warnings

"""
suppress warnings:
UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  warnings.warn(
FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
"""

warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*")

"""
Create a brand-new, unique Kokoro voice from an existing one - not by blending voices together (see kokoro-blend-voices.py for that), but by moving a
voice through Kokoro's voice-embedding space along meaningful directions.

Background: a Kokoro voice is not a model - it is a (510, 1, 256) float32 style tensor. Row t holds the 256-dim style vector used when the input is
t+1 tokens long. Kokoro feeds dims [0:128] to its decoder (timbre / voice quality) and dims [128:256] to its prosody predictor (pitch, duration,
energy). Because of this, "a new person" is just vector arithmetic: measuring the official 54-voice pack shows two distinct voices sit 1.5-4.0 apart
(median ~2.8, L2 on per-voice means), so a delta of that size - applied uniformly to all 510 rows - reads as a different person, while a small delta
just sounds like the same person "moderately off". Past ~3.5 artifacts creep in (tinny, distorted), and by ~5.4 you get demons.

The knobs (all composable in a single run):
  --gender   +x toward feminine, -x toward masculine (axis: official female minus male English voices; norm ~1.2, so +-1.5 is person-scale)
  --age      +x toward elderly (axis from the three 'santa' voices), -x toward child (axis from zm_yunxia, the young-boy voice). The two axes are
             nearly orthogonal - old and young are different departures from "adult", not opposites. Unit-norm; +-1.0..2.0 is the useful range, and
             at +2.0 you are essentially AT santa, so stay lower to remain distinct.
  --british  +x toward British English, -x toward American (per-gender-averaged axis between the b* and a* voices)
  --alpha    extrapolation: out = ref + alpha * (base - ref). 0..1 is ordinary blending; >1 exaggerates whatever makes base distinct from --ref
             (default ref: the average of all voices). 1.5..2.5 is the useful range.
  --strength random perturbation sampled from the pack's top principal components, scaled by per-PC std - this stays on the "plausible human
             voice" manifold, so each --seed is a different believable stranger. 1.0..1.4 is the useful range.

  --target-distance  after combining the knobs above, rescale the total delta to this L2 norm, so you can think in "how different a person" units:
                     ~1.0 = same person slightly off, ~2.5 = clearly a new person (default when knobs would otherwise underdo it), ~3.5 = ceiling
                     before artifacts. Set 0 to disable rescaling and use the raw knob values.

This script produces two files (like kokoro-blend-voices.py):
* A .pt file - loadable by Kokoro exactly like an official voice.
* A .wav file - a spoken sample of the new voice (skipped if the kokoro package is not installed, e.g. on a box without the TTS environment;
  pass --voices-dir with a folder of official voice .pt files to run there).

Examples:
    # a believable new stranger near af_heart (different seed = different stranger)
    python kokoro-make-voice.py --base af_heart --strength 1.2 --seed 42

    # a young girl derived from af_heart
    python kokoro-make-voice.py --base af_heart --gender 0.5 --age -1.2 --file-name "young_girl"

    # an older, more masculine, slightly British af_bella, scaled to "clearly a new person"
    python kokoro-make-voice.py --base af_bella --gender -0.8 --age 1.0 --british 1.0 --target-distance 2.5
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

try:
    import soundfile as sf
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

REPO_ID = 'hexgrad/Kokoro-82M'
LANGUAGE_CODE = 'a'
OUTPUT_DIR = '.'
SAMPLE_RATE = 24000
N_ROWS, N_DIMS = 510, 256
TOP_PCS = 24  # covers ~90% of the between-voice variance of the official pack
SAMPLE_TEXT = ("Well, hello there - I am a brand new voice, and nobody has ever heard me before today. "
               "I hope you like how I sound.")

# The default voices in the hexgrad/Kokoro-82M repo. The embedding-space axes (gender, age, accent, PCA)
# are measured from these, so the full set should be available either via kokoro or --voices-dir.
COMMON_VOICES = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah',
    'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice',
    'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 'em_santa',
    'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi',
    'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian',
    'zm_yunxi', 'zm_yunxia', 'zm_yunyang']


def load_voices(language_code: str, repo_id: str, voices_dir: str = None):
    """
    Load the official Kokoro voices as numpy arrays, via the kokoro package if available (downloading from HuggingFace
    as needed) or from a local directory of .pt files.

    Args:
        language_code: the language code as defined by Kokoro; only used to build the pipeline for the sample .wav.
        repo_id: The repo ID where the voices exist in HuggingFace. The default is hexgrad/Kokoro-82M.
        voices_dir: optional local directory of official voice .pt files; required when kokoro is not installed.
    Returns:
        A Tuple - (KPipeline or None, a dictionary of voice name -> (510, 256) numpy array)
    """
    voice_data = {}

    if voices_dir:
        print(f"Loading voices from {voices_dir}...")
        for path in sorted(glob.glob(os.path.join(voices_dir, "*.pt"))):
            name = os.path.basename(path)[:-3]
            voice_data[name] = torch.load(path, weights_only=True).squeeze(1).cpu().numpy()
        pipeline = KPipeline(lang_code=language_code, repo_id=repo_id) if KOKORO_AVAILABLE else None
        return pipeline, voice_data

    if not KOKORO_AVAILABLE:
        raise SystemExit("The kokoro package is not installed - pass --voices-dir with a folder of official voice .pt files instead.")

    print("Loading Kokoro voices...")
    pipeline = KPipeline(lang_code=language_code, repo_id=repo_id)
    for voice_name in COMMON_VOICES:
        voice_tensor = pipeline.load_voice(voice_name)
        if voice_tensor is not None:
            voice_data[voice_name] = voice_tensor.squeeze(1).cpu().numpy()

    return pipeline, voice_data


def voice_axes(voice_data: dict):
    """
    Measure the meaningful directions of the voice-embedding space from the official pack.

    Args:
        voice_data: A dictionary of voice name -> (510, 256) numpy array; should contain the full official pack.
    Returns:
        A dict with: 'means' (per-voice mean vectors), 'names', 'gender' (masc->fem axis), 'elderly' and 'youth'
        (unit-norm age axes), 'british' (American->British axis), 'pcs' (top principal components), 'pc_stds'.
    """
    names = list(voice_data)
    means = np.stack([v.mean(axis=0) for v in voice_data.values()])
    by = dict(zip(names, means))

    def group_mean(prefixes):
        rows = [by[n] for n in names if any(n.startswith(p) for p in prefixes)]
        return np.mean(rows, axis=0)

    gender = group_mean(['af_', 'bf_']) - group_mean(['am_', 'bm_'])
    british = ((group_mean(['bf_']) - group_mean(['af_'])) + (group_mean(['bm_']) - group_mean(['am_']))) / 2

    # Age: the 'santa' voices are elderly men; zm_yunxia is a young boy. The two directions are nearly
    # orthogonal (cos ~ +0.1) - old and young are different departures from "adult" - so each gets its own axis.
    am_adults = [by[n] for n in names if n.startswith('am_') and n != 'am_santa']
    elderly = np.mean([by['am_santa'] - np.mean(am_adults, axis=0),
                       by['em_santa'] - by['em_alex'],
                       by['pm_santa'] - by['pm_alex']], axis=0)
    youth = by['zm_yunxia'] - np.mean([by['zm_yunjian'], by['zm_yunxi'], by['zm_yunyang']], axis=0)
    elderly /= np.linalg.norm(elderly)
    youth /= np.linalg.norm(youth)

    centered = means - means.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc_stds = S / np.sqrt(len(names) - 1)

    return {'means': means, 'names': names, 'gender': gender, 'elderly': elderly,
            'youth': youth, 'british': british, 'pcs': Vt, 'pc_stds': pc_stds}


def build_voice(voice_data: dict, axes: dict, args) -> tuple:
    """
    Combine the requested knobs into a single delta and apply it to the base voice.

    Args:
        voice_data: A dictionary of voice name -> (510, 256) numpy array.
        axes: the measured axes from voice_axes().
        args: the parsed CLI arguments (base, gender, age, british, alpha, ref, strength, seed, target_distance).
    Returns:
        A Tuple - (the new (510, 256) float32 numpy array, a list of short tags describing what was done)
    """
    base = voice_data[args.base]
    delta = np.zeros(N_DIMS, dtype=np.float64)
    tags = []

    if args.alpha != 1.0:
        ref = voice_data[args.ref].mean(axis=0) if args.ref else axes['means'].mean(axis=0)
        delta += (args.alpha - 1.0) * (base.mean(axis=0) - ref)
        tags.append(f"a{args.alpha:g}" + (f"-{args.ref}" if args.ref else ""))

    if args.gender != 0.0:
        delta += args.gender * axes['gender']
        tags.append(f"g{args.gender:+g}")

    if args.age != 0.0:
        # positive -> elderly direction, negative -> youth direction (separate, near-orthogonal axes)
        delta += abs(args.age) * (axes['elderly'] if args.age > 0 else axes['youth'])
        tags.append(f"y{args.age:+g}")

    if args.british != 0.0:
        delta += args.british * axes['british']
        tags.append(f"b{args.british:+g}")

    if args.strength != 0.0:
        rng = np.random.default_rng(args.seed)
        coeffs = rng.standard_normal(TOP_PCS) * axes['pc_stds'][:TOP_PCS]
        delta += args.strength * (coeffs @ axes['pcs'][:TOP_PCS])
        tags.append(f"s{args.strength:g}-seed{args.seed}")

    norm = np.linalg.norm(delta)
    if args.target_distance > 0 and norm > 0:
        delta *= args.target_distance / norm
        norm = args.target_distance
        tags.append(f"d{args.target_distance:g}")

    print(f"Delta norm: {norm:.2f}  (~1 = same person slightly off, ~2.5 = new person, >3.5 = artifact territory)")
    return (base + delta[None, :]).astype(np.float32), tags


def save_voice(arr: np.ndarray, pipeline, output_name: str) -> str:
    """
    Save the new voice as a .pt and, if kokoro is available, render a sample .wav of it.

    Args:
        arr: the new (510, 256) float32 numpy array.
        pipeline: The Kokoro pipeline, or None if kokoro is unavailable (skips the sample).
        output_name: The name of the output file. Do not include an extension - this is used for both .pt and .wav.
    Returns:
        The name of the .pt file generated.
    """
    tensor = torch.from_numpy(arr).unsqueeze(1)  # (510, 1, 256), same layout as the official voices
    assert tensor.shape == (N_ROWS, 1, N_DIMS) and tensor.dtype == torch.float32

    filename = f"{output_name}.pt"
    torch.save(tensor, filename)
    print(f"PyTorch Pickle File: {filename}")

    if pipeline is None:
        print("kokoro not installed - skipping the sample .wav (the .pt is still fully usable).")
        return filename

    try:
        test_voice_name = str(output_name)
        reference = pipeline.load_voice(COMMON_VOICES[0])
        pipeline.voices[test_voice_name] = tensor.to(reference.device, reference.dtype)

        audio_chunks = [audio for _, _, audio in pipeline(SAMPLE_TEXT, voice=test_voice_name) if audio is not None]
        if audio_chunks:
            test_filename = f"{output_name}.wav"
            sf.write(test_filename, np.concatenate(audio_chunks), SAMPLE_RATE)
            print(f"Sample audio: {test_filename}")
        else:
            print("No sample audio generated")

        del pipeline.voices[test_voice_name]
    except Exception as e:
        print(f"Sample rendering failed ({e}) - the .pt is still fully usable.")

    return filename


def get_args_dict():
    """
    Parses the command-line arguments for the Kokoro voice-maker script.
    """
    parser = argparse.ArgumentParser(description='Kokoro Voice Maker - create new, unique voices by moving an existing voice through embedding space.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='The voice to start from, e.g. af_heart.')
    parser.add_argument('--gender', type=float, default=0.0, help='Shift toward feminine (+) or masculine (-); +-1.5 is person-scale.')
    parser.add_argument('--age', type=float, default=0.0, help='Shift toward elderly (+) or child (-); useful range +-1.0..2.0 (at +2.0 you basically ARE santa).')
    parser.add_argument('--british', type=float, default=0.0, help='Shift toward British (+) or American (-) English.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Extrapolation factor; >1 exaggerates what makes --base distinct from --ref. Useful range 1.5..2.5.')
    parser.add_argument('--ref', default=None, help='Reference voice for --alpha (default: the average of all voices).')
    parser.add_argument('--strength', type=float, default=0.0, help='Random perturbation along the voice manifold; useful range 1.0..1.4.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for --strength; each seed is a different believable stranger.')
    parser.add_argument('--target-distance', type=float, default=0.0, help='Rescale the combined delta to this L2 norm (~2.5 = clearly a new person, ~3.5 = ceiling). 0 = use raw knob values.')
    parser.add_argument('--file-name', default='', help='The base of the output filenames (.pt and .wav). Do not include an extension.')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Where to write the output files.')
    parser.add_argument('--voices-dir', default=None, help='Local directory of official voice .pt files; required if the kokoro package is not installed.')
    parser.add_argument('--language-code', default=LANGUAGE_CODE, help='The language code; a = American English, British English = b, Spanish = e, French = f, Italian = i, Brazilian Portuguese = p, Hindi = h.')
    parser.add_argument('--repo-id', default=REPO_ID, help="The repo ID - the vast majority of people use 'hexgrad/Kokoro-82M'.")
    return parser.parse_args()


def main():
    args = get_args_dict()

    pipeline, voice_data = load_voices(args.language_code, args.repo_id, args.voices_dir)
    if args.base not in voice_data:
        print(f"Voice '{args.base}' not found; available: {', '.join(voice_data)}")
        return
    if args.ref and args.ref not in voice_data:
        print(f"Reference voice '{args.ref}' not found; available: {', '.join(voice_data)}")
        return

    axes = voice_axes(voice_data)
    new_voice, tags = build_voice(voice_data, axes, args)

    if not tags:
        print("No knobs set - the output would be an exact copy of the base voice. See --help for the knobs.")
        return

    filename_no_ext = args.file_name if args.file_name else f"{args.base}__{'_'.join(tags)}"
    save_voice(new_voice, pipeline, Path(os.path.expanduser(args.output_dir)) / filename_no_ext)


if __name__ == "__main__":
    main()
