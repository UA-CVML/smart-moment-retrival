# Upstream QVHighlights evaluator

`eval.py` and `utils.py` are retained from the
[Moment-DETR evaluator](https://github.com/jayleicn/moment_detr/tree/main/standalone_eval).
They expect the official benchmark JSONL format; SMART's saved internal prediction
JSON is a different format. Consult the upstream instructions before using these
utilities or submitting predictions to the benchmark service.

For local SMART validation, use `bash scripts/eval_qvhighlights.sh` from the root
README. Historical sample predictions and generated metric files are not part of
this source release.
