import torch

from . import vte_core as core
from .vte_core import VTE_TYPE, VTE_INPUT_TYPE

CATEGORY = "video_to_extend"
BIG = 0xFFFFFFFF


class VTENull:
    """Makes a null VideoToExtend containing only default values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (VTE_TYPE,)
    RETURN_NAMES = ("vte",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self):
        return (core.default_vte(),)


class VTECreate:
    """Makes a non-null VideoToExtend from a video in ComfyUI's input folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # Same file list and upload option as VHS "Load Video (Upload)".
            "video": (core.list_input_videos(), {"video_upload": True}),
            "prefix_to_sustain": ("INT", {"default": 0, "min": 0, "max": BIG}),
            "working_frame_count": ("INT", {
                "default": -1, "min": -1, "max": BIG,
                "tooltip": "-1 = all frames after the prefix"}),
        }}

    RETURN_TYPES = (VTE_TYPE,)
    RETURN_NAMES = ("vte",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, video=None, **kwargs):
        # Re-run downstream nodes when the file itself changes on disk.
        try:
            return core.file_signature(core.resolve_video_path(video))
        except Exception:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, video=None, prefix_to_sustain=None,
                        working_frame_count=None):
        # Runs before the workflow starts, so bad values stop it immediately.
        # Values arriving through links are checked again in make().
        if video is None:
            return True
        try:
            path = core.resolve_video_path(video)
            if prefix_to_sustain is not None and working_frame_count is not None:
                total = core.probe_video(path)["frames"]
                core.check_counts(int(prefix_to_sustain),
                                  int(working_frame_count), total, video)
        except Exception as e:
            return str(e)
        return True

    def make(self, video, prefix_to_sustain, working_frame_count):
        vte = {
            "video_file": video,
            "is_null": False,
            "prefix_to_sustain": int(prefix_to_sustain),
            "working_frame_count": int(working_frame_count),
        }
        core.inspect_vte(vte)  # raises (halting the workflow) if invalid
        return (vte,)


class VTERead:
    """Decodes the prefix and working frames a VideoToExtend describes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vte": (VTE_INPUT_TYPE,)}}

    RETURN_TYPES = ("BOOLEAN", "IMAGE", "INT", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("is_null", "prefix_frames", "prefix_count",
                    "working_frames", "working_count", "fps")
    FUNCTION = "read"
    CATEGORY = CATEGORY

    def read(self, vte):
        vte = core.normalize_vte(vte)
        if vte["is_null"]:
            return (True, core.empty_batch(), 0, core.empty_batch(), 0, 0.0)

        info = core.inspect_vte(vte)
        explicit = vte["working_frame_count"] != -1
        prefix, work = core.decode_sections(
            info["path"], info["prefix"],
            info["working"] if explicit else None,
            info["height"], info["width"])

        short = prefix.shape[0] < info["prefix"] or (
            explicit and work.shape[0] < info["working"])
        if short:
            raise ValueError(
                f"VideoToExtend: {vte['video_file']!r} decoded to only "
                f"{prefix.shape[0] + work.shape[0]} frames, fewer than the "
                f"{info['prefix'] + info['working']} requested.")
        return (False, prefix, prefix.shape[0], work, work.shape[0],
                info["fps"])


class VTESplice:
    """Outputs the VTE's prefix followed by new frames, with an optional
    crossfade from the old working frames into the new ones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vte": (VTE_INPUT_TYPE,),
            "images": ("IMAGE",),
            "fade_frames": ("INT", {
                "default": 0, "min": 0, "max": BIG,
                "tooltip": "0 = no fade. Clamped silently to the frames "
                           "available on both sides."}),
            "fade_envelope": (list(core.FADE_ENVELOPES.keys()),),
            "fade_color_space": (list(core.FADE_COLOR_SPACES), {
                "tooltip": "Colour space the crossfade is mixed in. oklab "
                           "usually looks smoothest."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "splice"
    CATEGORY = CATEGORY

    def splice(self, vte, images, fade_frames, fade_envelope,
               fade_color_space):
        vte = core.normalize_vte(vte)
        if vte["is_null"]:
            return (images, images.shape[0])

        info = core.inspect_vte(vte)  # validates counts, cheap
        h, w = info["height"], info["width"]
        n_new = images.shape[0]

        # Size check happens before any decoding.
        if n_new > 0 and tuple(images.shape[1:]) != (h, w, 3):
            raise ValueError(
                "VideoToExtend Splice: input images are "
                f"{images.shape[2]}x{images.shape[1]} with {images.shape[3]} "
                f"channels, but {vte['video_file']!r} is {w}x{h} RGB.")

        n_fade = max(0, min(int(fade_frames), info["working"], n_new))
        prefix, old = core.decode_sections(
            info["path"], info["prefix"], n_fade, h, w)
        if prefix.shape[0] < info["prefix"]:
            raise ValueError(
                f"VideoToExtend: {vte['video_file']!r} decoded to only "
                f"{prefix.shape[0]} frames, fewer than the prefix "
                f"({info['prefix']}).")
        n_fade = min(n_fade, old.shape[0])  # graceful if decoding fell short

        device = images.device
        dtype = images.dtype if n_new > 0 else torch.float32
        parts = [prefix.to(device, dtype)]
        if n_fade > 0:
            parts.append(core.blend_frames(
                old[:n_fade].to(device, dtype), images[:n_fade],
                fade_color_space, fade_envelope))
        if n_new > n_fade:
            parts.append(images[n_fade:])
        out = torch.cat(parts, dim=0)
        return (out, out.shape[0])


class VTEToDict:
    """Explicitly turns a VideoToExtend into a plain DICT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vte": (VTE_INPUT_TYPE,)}}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("dict",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, vte):
        return (core.normalize_vte(vte),)


class DictToVTE:
    """Treats a DICT as a VideoToExtend (missing fields get defaults)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dict": ("DICT",)}}

    RETURN_TYPES = (VTE_TYPE,)
    RETURN_NAMES = ("vte",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, dict):
        return (core.normalize_vte(dict),)


NODE_CLASS_MAPPINGS = {
    "VTE_Null": VTENull,
    "VTE_Create": VTECreate,
    "VTE_Read": VTERead,
    "VTE_Splice": VTESplice,
    "VTE_ToDict": VTEToDict,
    "VTE_FromDict": DictToVTE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTE_Null": "Video To Extend (Null)",
    "VTE_Create": "Video To Extend (Create)",
    "VTE_Read": "Video To Extend (Read)",
    "VTE_Splice": "Video To Extend (Splice Frames)",
    "VTE_ToDict": "Video To Extend → DICT",
    "VTE_FromDict": "DICT → Video To Extend",
}
