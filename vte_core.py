"""
Core helpers for VideoToExtend (VTE) objects.

A VTE is a plain Python dict with four fields:

    video_file           str | None  name as used by VHS "Load Video (Upload)"
                                      (relative to ComfyUI's input dir, may carry
                                      an annotation like "clip.mp4 [input]"),
                                      or an absolute path.
    is_null              bool        True -> nothing here may touch video_file.
    prefix_to_sustain    int  >= 0   frames at the start kept unchanged.
    working_frame_count  int  >= -1  frames after the prefix to work on;
                                      -1 means "all remaining frames".

Passing a VTE around is just passing that small dict, so it is cheap. Frames
are only decoded by nodes that actually output or need them, and those nodes'
outputs are cached by ComfyUI's normal execution cache.
"""

import os

import numpy as np
import torch

import folder_paths

VTE_TYPE = "VIDEO_TO_EXTEND"
# Inputs accept either the dedicated type or a plain DICT (comma = union type).
VTE_INPUT_TYPE = "VIDEO_TO_EXTEND,DICT"

# Same list VideoHelperSuite uses for its input-directory video picker.
VIDEO_EXTENSIONS = ("webm", "mp4", "mkv", "gif", "mov")

EMPTY_PLACEHOLDER_SIZE = 64  # H and W of empty batches when no video is known


def default_vte():
    return {
        "video_file": None,
        "is_null": True,
        "prefix_to_sustain": 0,
        "working_frame_count": -1,
    }


def list_input_videos():
    """Video files in the top level of ComfyUI's input directory (as VHS does)."""
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            parts = f.split(".")
            if len(parts) > 1 and parts[-1].lower() in VIDEO_EXTENSIONS:
                files.append(f)
    return sorted(files)


def normalize_vte(obj):
    """Return a clean copy of a VTE dict, filling defaults and checking types."""
    if obj is None:
        return default_vte()
    if not isinstance(obj, dict):
        raise ValueError(
            f"VideoToExtend: expected a dict, got {type(obj).__name__}.")
    vte = default_vte()
    vte.update(obj)
    try:
        vte["is_null"] = bool(vte["is_null"])
        vte["prefix_to_sustain"] = int(vte["prefix_to_sustain"])
        vte["working_frame_count"] = int(vte["working_frame_count"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"VideoToExtend: malformed field value ({e}).")
    if not vte["is_null"] and not vte["video_file"]:
        raise ValueError(
            "VideoToExtend: is_null is False but video_file is empty.")
    return vte


def resolve_video_path(video_file):
    name = str(video_file).strip().strip('"')
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    path = folder_paths.get_annotated_filepath(name)
    if path and os.path.isfile(path):
        return path
    raise ValueError(f"VideoToExtend: video file not found: {video_file!r}")


def file_signature(path):
    st = os.stat(path)
    return f"{path}|{st.st_mtime_ns}|{st.st_size}"


# --- probing (cheap: demux only, no decoding) --------------------------------

_probe_cache = {}
_PROBE_CACHE_MAX = 64


def probe_video(path):
    """Return dict(frames, width, height, fps). Cached per file version.

    The frame count comes from counting video packets, which avoids decoding
    and matches the decoded count for ordinary files. Decoding nodes still
    double-check and raise an error if the decoder produces fewer frames.
    """
    key = file_signature(path)
    hit = _probe_cache.get(key)
    if hit is not None:
        return hit
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        has_audio = len(container.streams.audio) > 0
        frames = 0
        for packet in container.demux(stream):
            if packet.size and not getattr(packet, "is_discard", False):
                frames += 1
    info = {"frames": frames, "width": width, "height": height, "fps": fps,
            "has_audio": has_audio}
    if len(_probe_cache) >= _PROBE_CACHE_MAX:
        _probe_cache.clear()
    _probe_cache[key] = info
    return info


def check_counts(prefix, working, total, video_name):
    """Validate the two counts against a video length; return resolved counts.

    Raises ValueError (which halts the workflow) on any problem.
    """
    if prefix < 0:
        raise ValueError(
            f"VideoToExtend: prefix_to_sustain must be >= 0 (got {prefix}).")
    if working < -1:
        raise ValueError(
            "VideoToExtend: working_frame_count must be >= -1 "
            f"(got {working}).")
    if prefix > total:
        raise ValueError(
            f"VideoToExtend: prefix_to_sustain ({prefix}) exceeds the "
            f"{total} frames in {video_name!r}.")
    if working == -1:
        working = total - prefix
    elif prefix + working > total:
        raise ValueError(
            f"VideoToExtend: prefix_to_sustain ({prefix}) + "
            f"working_frame_count ({working}) = {prefix + working} exceeds "
            f"the {total} frames in {video_name!r}.")
    return prefix, working


def inspect_vte(vte):
    """Resolve a non-null VTE: path, video info and validated counts."""
    path = resolve_video_path(vte["video_file"])
    info = probe_video(path)
    prefix, working = check_counts(
        vte["prefix_to_sustain"], vte["working_frame_count"],
        info["frames"], vte["video_file"])
    return dict(info, path=path, prefix=prefix, working=working)


# --- decoding -----------------------------------------------------------------

def _to_image_batch(arrays, height, width):
    if not arrays:
        return torch.zeros((0, height, width, 3), dtype=torch.float32)
    batch = torch.from_numpy(np.stack(arrays))
    arrays.clear()
    return batch.to(torch.float32).div_(255.0)


def decode_ranges(path, ranges, height, width):
    """Decode the frames in each [start, end) range in one sequential pass.

    end=None means "to the end of the file". Reading stops as soon as every
    range is complete, and only frames inside some range are converted.
    Returns one IMAGE batch (B, H, W, 3) per range; ranges may overlap.
    """
    ranges = [(max(0, int(a)), None if b is None else int(b)) for a, b in ranges]
    live = [(a, b) for a, b in ranges if b is None or b > a]
    kept = [[] for _ in ranges]
    if live:
        limit = (None if any(b is None for _, b in live)
                 else max(b for _, b in live))
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                arr = None
                for k, (a, b) in enumerate(ranges):
                    if i >= a and (b is None or i < b):
                        if arr is None:
                            arr = frame.to_ndarray(format="rgb24")
                        kept[k].append(arr)
                if limit is not None and i + 1 >= limit:
                    break
    return [_to_image_batch(frames, height, width) for frames in kept]


def decode_sections(path, n_prefix, n_work, height, width):
    """Decode frames [0, n_prefix) and [n_prefix, n_prefix + n_work).

    n_work=None decodes everything after the prefix.
    """
    end = None if n_work is None else n_prefix + n_work
    prefix, work = decode_ranges(
        path, [(0, n_prefix), (n_prefix, end)], height, width)
    return prefix, work


def empty_batch():
    s = EMPTY_PLACEHOLDER_SIZE
    return torch.zeros((0, s, s, 3), dtype=torch.float32)


# --- fading -------------------------------------------------------------------

FADE_ENVELOPES = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1.0 - (1.0 - t) ** 2,
    "ease_in_out": lambda t: t * t * (3.0 - 2.0 * t),  # smoothstep
}

# All options are plain crossfades; they differ only in the colour space the
# mix happens in, which changes how smooth the transition looks.
FADE_COLOR_SPACES = (
    "oklab",         # perceptually uniform: even-looking lightness and hue
    "linear_light",  # physically correct light mixing, like a film dissolve
    "srgb",          # naive mix of the stored values
)


def _srgb_to_linear(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c ** (1.0 / 2.4) - 0.055)


# OKLab matrices (Björn Ottosson), for linear sRGB input.
_RGB_TO_LMS = [[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]]
_LMS_TO_LAB = [[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]]


def _mat(m, ref):
    return torch.tensor(m, dtype=torch.float32, device=ref.device)


def _srgb_to_oklab(c):
    lms = _srgb_to_linear(c) @ _mat(_RGB_TO_LMS, c).T
    return torch.sign(lms) * lms.abs() ** (1.0 / 3.0) @ _mat(_LMS_TO_LAB, c).T


def _oklab_to_srgb(lab):
    lms = (lab @ torch.linalg.inv(_mat(_LMS_TO_LAB, lab)).T) ** 3
    return _linear_to_srgb(lms @ torch.linalg.inv(_mat(_RGB_TO_LMS, lab)).T)


_SPACES = {
    "srgb": (lambda c: c, lambda c: c),
    "linear_light": (_srgb_to_linear, _linear_to_srgb),
    "oklab": (_srgb_to_oklab, _oklab_to_srgb),
}


def blend_frames(old, new, color_space, envelope):
    """Crossfade from `old` to `new` (same shape, N frames).

    Frame i uses t = (i + 1) / (N + 1), so the fade never shows a pure old or
    pure new frame at its ends; the frames before/after supply those.
    """
    n = old.shape[0]
    t = torch.arange(1, n + 1, dtype=torch.float32) / (n + 1)
    e = FADE_ENVELOPES[envelope](t).to(new.device).view(n, 1, 1, 1)
    to_space, from_space = _SPACES[color_space]
    a = to_space(old.float())
    b = to_space(new.float())
    return from_space(a + (b - a) * e).clamp(0.0, 1.0).to(new.dtype)


# --- audio --------------------------------------------------------------------
#
# Audio values use ComfyUI's AUDIO format:
#     {"waveform": tensor (batch, channels, samples), "sample_rate": int}
# Internally a "track" is (waveform (channels, samples) float32, sample_rate),
# or None when a source is missing or empty.

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
NULL_VTE_FPS = 24.0  # only used to size silence when no video gives a rate


def audio_from_input(audio):
    """Turn an AUDIO input into a track, or None if missing or empty."""
    if not isinstance(audio, dict):
        return None
    wave, sr = audio.get("waveform"), audio.get("sample_rate")
    if wave is None or not sr:
        return None
    wave = torch.as_tensor(wave).detach().cpu().float()
    if wave.dim() == 3:
        wave = wave[0]
    elif wave.dim() == 1:
        wave = wave[None]
    if wave.dim() != 2 or wave.shape[0] == 0 or wave.shape[1] == 0:
        return None
    return wave, int(sr)


def _stream_start(stream):
    if stream is None or stream.start_time is None or stream.time_base is None:
        return 0.0
    return float(stream.start_time * stream.time_base)


def decode_audio(path, seconds):
    """First `seconds` of the file's audio, aligned to its first video frame.

    Returns a track, or None if the file has no audio or none in that span.
    Decoding problems are treated as missing audio rather than errors.
    """
    if seconds <= 0:
        return None
    try:
        import av
        with av.open(path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            video = container.streams.video[0] if container.streams.video else None
            sr = int(stream.codec_context.sample_rate or stream.rate or 0)
            if sr <= 0:
                return None
            offset = round((_stream_start(stream) - _stream_start(video)) * sr)
            wanted = round(seconds * sr)
            need = wanted - offset  # samples to read from the stream itself
            resampler = av.AudioResampler(format="fltp", rate=sr)
            chunks, got = [], 0
            if need > 0:
                for frame in container.decode(stream):
                    for f in resampler.resample(frame):
                        arr = f.to_ndarray()
                        chunks.append(arr)
                        got += arr.shape[1]
                    if got >= need:
                        break
    except Exception as e:
        print(f"[comfyui_video_to_extend] could not read audio of {path!r}: {e}")
        return None
    if not chunks:
        return None
    wave = torch.from_numpy(np.concatenate(chunks, axis=1)).float()
    if offset > 0:
        wave = torch.cat([torch.zeros(wave.shape[0], offset), wave], dim=1)
    elif offset < 0:
        wave = wave[:, -offset:]
    wave = wave[:, :wanted]
    return (wave, sr) if wave.shape[1] > 0 else None


def _resample(wave, sr, target):
    if sr == target:
        return wave
    try:
        import torchaudio.functional as AF
        return AF.resample(wave, sr, target)
    except Exception:
        n = max(1, round(wave.shape[1] * target / sr))
        return torch.nn.functional.interpolate(
            wave[None], size=n, mode="linear", align_corners=False)[0]


def _conform(track, sr, channels):
    if track is None:
        return None
    wave = _resample(track[0], track[1], sr)
    c = wave.shape[0]
    if c < channels:
        wave = wave.repeat((channels + c - 1) // c, 1)
    return wave[:channels]


def silent_audio(seconds, sr=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS):
    n = max(1, round(seconds * sr))
    return {"waveform": torch.zeros(1, channels, n), "sample_rate": sr}


def _ramp(n, envelope):
    """Rising fade weights for n samples, never exactly 0 or 1 (no clicks)."""
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    return FADE_ENVELOPES[envelope](t)


def mix_audio(orig, new, start_frame, new_frames, total_frames, fps,
              fade_in_frames, fade_out_frames, envelope):
    """Audio for a splice, laid out on the output video's timeline.

    The output keeps the original video's timing outside the new section, so
    the original track plays at its own timestamps. The new track plays under
    the new section [start_frame, start_frame + new_frames), from its own
    beginning, trimmed or padded to fit. Inside that section it crossfades in
    from the original over the first fade_in_frames and back out to the
    original over the last fade_out_frames, using the video's envelope.
    Missing or short audio is silence. Output length matches total_frames.
    """
    sr = new[1] if new else (orig[1] if orig else DEFAULT_SAMPLE_RATE)
    channels = max([t[0].shape[0] for t in (orig, new) if t]
                   or [DEFAULT_CHANNELS])
    o, n = _conform(orig, sr, channels), _conform(new, sr, channels)

    def at(frame):
        return round(frame * sr / fps)

    s_total = max(1, at(total_frames))
    out = torch.zeros(channels, s_total)
    if o is not None:
        k = min(o.shape[1], s_total)
        out[:, :k] = o[:, :k]

    s0 = min(at(start_frame), s_total)
    s1 = min(at(start_frame + new_frames), s_total)
    if s1 > s0:
        length = s1 - s0
        region = torch.zeros(channels, length)
        if n is not None:
            k = min(n.shape[1], length)
            region[:, :k] = n[:, :k]
        weight = torch.ones(length)
        f_in = min(at(start_frame + fade_in_frames) - s0, length)
        f_out = min(s1 - at(start_frame + new_frames - fade_out_frames), length)
        if f_in > 0:
            weight[:f_in] = _ramp(f_in, envelope)
        if f_out > 0:
            weight[length - f_out:] = torch.minimum(
                weight[length - f_out:], 1.0 - _ramp(f_out, envelope))
        old = out[:, s0:s1].clone()
        out[:, s0:s1] = old * (1.0 - weight) + region * weight
    return {"waveform": out[None], "sample_rate": sr}
