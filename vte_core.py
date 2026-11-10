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
        frames = 0
        for packet in container.demux(stream):
            if packet.size and not getattr(packet, "is_discard", False):
                frames += 1
    info = {"frames": frames, "width": width, "height": height, "fps": fps}
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


def decode_sections(path, n_prefix, n_work, height, width):
    """Decode frames [0, n_prefix) and [n_prefix, n_prefix + n_work).

    n_work=None decodes everything after the prefix. Stops reading as soon as
    the requested frames are in hand. Returns two IMAGE batches (B, H, W, 3).
    """
    prefix, work = [], []
    limit = None if n_work is None else n_prefix + n_work
    if limit != 0:
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            i = 0
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                (prefix if i < n_prefix else work).append(arr)
                i += 1
                if limit is not None and i >= limit:
                    break
    return (_to_image_batch(prefix, height, width),
            _to_image_batch(work, height, width))


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
