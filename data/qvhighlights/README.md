# QVHighlights data used by this release

Dataset source: [QVHighlights / Moment-DETR](https://github.com/jayleicn/moment_detr).
The included annotation files are used by the training and evaluation configs.

| File | Queries | Use |
| --- | ---: | --- |
| `train.json` | 7,218 | Training |
| `val.json` | 1,550 | Local validation with real moment labels |
| `test_dummy.json` | 1,542 | Inference only; windows are dummy placeholders |

Each file is a JSON array. Entries contain `video` (clip basename without `.mp4`),
`qid` (for example `QVHighlight_2579`), `query`, `duration` in seconds, and
`relevant_windows` as a list of `[start, end]` pairs in seconds. The qid prefix is
internal to this implementation; official submissions use numeric query IDs.

`scene_boundaries.json` provides shot boundaries for QVHighlights. It maps `<video_id>.mp4` to `scene_start_times` (seconds) and
`scene_start_frames` (frame indices). It covers 10,147 of the 10,148 distinct
annotated clips. `MxM1yP8cRD4_210.0_360.0.mp4` has no cached entry and uses the
uniform 3-second segmentation fallback. Other missing or empty boundary
entries also use this fallback.

The paper uses TransNetV2 shot boundaries. Use the supplied boundaries for these
clips; the shot detector is not included in this repository.
For replacement videos, regenerate boundary times for the exact clips being used.

Download the corresponding raw clips, including audio, into `videos/` or override
`datasets.qvh.build_info.videos.storage`. Video files, model weights and evaluation
outputs are excluded from Git. Dataset terms remain those of the original source.
