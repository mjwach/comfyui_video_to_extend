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
    crossfade from the old working frames into the new ones, plus a matching
    spliced audio track. With return_to_original on, the new frames are also
    faded back out into the original, which then continues to its end."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vte": (VTE_INPUT_TYPE,),
                "images": ("IMAGE",),
                "fade_frames": ("INT", {
                    "default": 0, "min": 0, "max": BIG,
                    "tooltip": "0 = no fade. Clamped silently to the frames "
                               "available. Also used for the audio "
                               "crossfade, and for the fade back out when "
                               "return_to_original is on."}),
                "fade_envelope": (list(core.FADE_ENVELOPES.keys()),),
                "fade_color_space": (list(core.FADE_COLOR_SPACES), {
                    "tooltip": "Colour space the crossfade is mixed in. oklab "
                               "usually looks smoothest."}),
                "return_to_original": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: the output ends with the new frames. "
                               "On: the new frames and the working section "
                               "are both cut to the shorter of the two, the "
                               "new frames fade in and back out, and the "
                               "rest of the original video follows, so the "
                               "output is as long as the original."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio track for the input images, starting at "
                               "their first frame."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN")
    RETURN_NAMES = ("images", "audio", "has_audio")
    FUNCTION = "splice"
    CATEGORY = CATEGORY

    def splice(self, vte, images, fade_frames, fade_envelope,
               fade_color_space, return_to_original=False, audio=None):
        vte = core.normalize_vte(vte)
        new_track = core.audio_from_input(audio)
        # A connected AUDIO counts as a track even if it's silent or empty.
        new_has_track = (isinstance(audio, dict)
                         and audio.get("waveform") is not None)
        n_new = images.shape[0]

        if vte["is_null"]:
            # No video, so no frame rate to align to: pass audio through.
            if new_track is not None:
                out_audio = {"waveform": new_track[0][None],
                             "sample_rate": new_track[1]}
            else:
                out_audio = core.silent_audio(n_new / core.NULL_VTE_FPS)
            return (images, out_audio, new_has_track)

        info = core.inspect_vte(vte)  # validates counts, cheap
        h, w = info["height"], info["width"]

        # Size check happens before any decoding.
        if n_new > 0 and tuple(images.shape[1:]) != (h, w, 3):
            raise ValueError(
                "VideoToExtend Splice: input images are "
                f"{images.shape[2]}x{images.shape[1]} with {images.shape[3]} "
                f"channels, but {vte['video_file']!r} is {w}x{h} RGB.")

        prefix_n, working = info["prefix"], info["working"]
        fade = max(0, int(fade_frames))
        if return_to_original:
            k = min(n_new, working)        # new frames and working set, cut
            new = images[:k]               # to the same length
            fade = min(fade, k // 2)       # the two fades at most meet
            ranges = [(0, prefix_n),                       # prefix
                      (prefix_n, prefix_n + fade),         # under fade-in
                      (prefix_n + k - fade, prefix_n + k),  # under fade-out
                      (prefix_n + k, None)]                # rest of video
        else:
            k = n_new
            new = images
            fade = min(fade, working, n_new)
            ranges = [(0, prefix_n), (prefix_n, prefix_n + fade),
                      (0, 0), (0, 0)]
        prefix, old_in, old_out, tail = core.decode_ranges(
            info["path"], ranges, h, w)
        if prefix.shape[0] < prefix_n:
            raise ValueError(
                f"VideoToExtend: {vte['video_file']!r} decoded to only "
                f"{prefix.shape[0]} frames, fewer than the prefix "
                f"({prefix_n}).")
        # Graceful if the decoder came up short of the probed frame count.
        fade_in = min(fade, old_in.shape[0])
        fade_out = fade if return_to_original else 0
        if old_out.shape[0] != fade_out:
            fade_out = 0

        device = images.device
        dtype = images.dtype if n_new > 0 else torch.float32

        def put(t):
            return t.to(device, dtype)

        parts = [put(prefix)]
        if fade_in > 0:
            parts.append(core.blend_frames(
                put(old_in[:fade_in]), new[:fade_in],
                fade_color_space, fade_envelope))
        if k - fade_out > fade_in:
            parts.append(new[fade_in:k - fade_out])
        if fade_out > 0:
            parts.append(core.blend_frames(
                new[k - fade_out:k], put(old_out),
                fade_color_space, fade_envelope))
        if tail.shape[0] > 0:
            parts.append(put(tail))
        out = torch.cat(parts, dim=0)
        total = out.shape[0]

        fps = info["fps"] if info["fps"] > 0 else core.NULL_VTE_FPS
        file_has_track = info.get("has_audio", True)
        orig_frames = total if return_to_original else prefix_n + fade_in
        orig_track = (core.decode_audio(info["path"], orig_frames / fps)
                      if file_has_track else None)
        out_audio = core.mix_audio(
            orig_track, new_track, prefix_n, k, total, fps,
            fade_in, fade_out, fade_envelope)
        # A source counts if it has a track (even a silent one) and some of
        # the output's timeline comes from it.
        orig_used = total > k or fade_in + fade_out > 0
        has_audio = ((file_has_track and orig_used)
                     or (new_has_track and k > 0))
        return (out, out_audio, has_audio)


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
