# comfyui_video_to_extend

ComfyUI nodes for working with **VideoToExtend** (VTE) objects: lightweight
descriptions of a video that is to be extended or partly regenerated.

A VTE is a small dict:

| Field                 | Type        | Default | Meaning |
|-----------------------|-------------|---------|---------|
| `video_file`          | str / null  | null    | Video name as used by VideoHelperSuite's *Load Video (Upload)* (relative to ComfyUI's `input` folder, annotations like `clip.mp4 [input]` allowed), or an absolute path. |
| `is_null`             | bool        | true    | When true, nothing reads `video_file`. |
| `prefix_to_sustain`   | int ≥ 0     | 0       | Frames at the start that are kept unchanged. |
| `working_frame_count` | int ≥ -1    | -1      | Frames after the prefix to work on. `-1` = all remaining frames. Frames after this section are discarded. |

Wires carry the type `VIDEO_TO_EXTEND`, so they won't connect to unrelated
sockets. Every node input that takes a VTE also accepts a plain `DICT`.

## Nodes

All nodes are under the **video_to_extend** category.

- **Video To Extend (Null)** – no inputs; outputs a null VTE with default values.
- **Video To Extend (Create)** – pick a video from the input folder (same list
  and upload button as VHS *Load Video (Upload)*) plus the two frame counts.
  Always outputs a non-null VTE. A `show_preview` toggle (off by default)
  shows or hides ComfyUI's video preview on the node. The toggle is
  frontend-only: it's saved with the workflow but never sent to the backend,
  so flipping it doesn't make downstream nodes re-run.
- **Video To Extend (Read)** – outputs `is_null`, the prefix frames, the prefix
  length, the working frames, their count, and the video's `fps` (handy for a
  Video Combine node). A null VTE gives empty batches and zero counts.
- **Video To Extend (Splice Frames)** – takes a VTE and an image batch and
  outputs *prefix + new frames*. The new batch may be shorter or longer than the
  working section it replaces, and any part may be empty. Optional crossfade
  from the old working frames into the new ones, meant to be as unnoticeable as
  possible:
  - `fade_frames` – 0 disables fading. Larger values are silently clamped to
    the frames actually available on both sides (no errors, no warnings).
  - `fade_envelope` – `linear`, `ease_in`, `ease_out`, `ease_in_out`.
  - `fade_color_space` – the colour space the mix happens in:
    - `oklab` (default) – perceptually uniform, so lightness and hue change
      evenly and mid-fade frames don't look washed out or muddy.
    - `linear_light` – mixes actual light intensities, like an optical film
      dissolve; physically correct, though dark areas can change unevenly.
    - `srgb` – plain mix of the stored pixel values; mid-fade frames tend to
      dip slightly darker and duller.

  Audio: an optional `audio` input is the soundtrack of the input images,
  starting at their first frame. The `audio` output is spliced to match the
  video: the video file's own audio under the prefix, then the input audio
  under the new frames, crossfading from the file's old working-section audio
  over the same `fade_frames` with the same `fade_envelope`. The input audio
  is trimmed or padded to exactly the new frames' duration. Any missing piece
  (no input audio, no audio track in the file, a track that ends early) is
  treated as silence, and if the two sources differ in sample rate or channel
  count, the input audio's sample rate and the larger channel count win.

  Outputs: `images`, `audio` (always a track, silent if need be), and
  `has_audio`, which is false when neither source contributed any audio, so a
  downstream node can choose to leave the audio track out entirely.

  With a null VTE the input images and audio pass through unchanged (silence,
  sized at 24 fps, if there's no input audio).
- **Video To Extend → DICT** / **DICT → Video To Extend** – explicit
  conversions. Missing dict fields get defaults.

## Errors

These stop the workflow:

- Frame counts that don't fit the video (`prefix_to_sustain` larger than the
  video, or `prefix + working_frame_count` larger than it) or negative values.
  When the Create node's values are typed into its widgets, this is caught
  before the workflow even starts; otherwise it's caught when the Create, Read
  or Splice node runs.
- A missing video file.
- Splice input images whose width, height or channel count differs from the
  video. This is checked before any video decoding.

## Performance notes

- A VTE is a dict of four small values, so passing it around costs nothing;
  no frames are attached to it.
- Frames are decoded only by **Read** and **Splice**, and only as far into the
  file as needed. Splice decodes just the prefix plus the frames it fades from.
- Validation counts frames by reading packets without decoding, and the result
  is cached per file version (path, modification time, size).
- ComfyUI's execution cache keeps node outputs, so a Read node isn't re-run
  unless its VTE or the video file changes.

## Installation

Clone or copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
It needs PyAV (`av`), which current ComfyUI already requires; otherwise run
`pip install -r requirements.txt` in ComfyUI's Python environment.

## License

Public domain, via [the Unlicense](LICENSE).
