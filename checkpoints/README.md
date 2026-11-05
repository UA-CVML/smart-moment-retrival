# Model files

Weights are kept outside Git. This release includes source code and data metadata.

| Component | Default source / path |
| --- | --- |
| Audio encoder | Download pretrained `BEATs_iter3` from [BEATs](https://github.com/microsoft/unilm/tree/master/beats) to `checkpoints/BEATs_iter3.pt` |
| Language model | `google/flan-t5-xl`, loaded by Hugging Face |
| Q-Former initialization | `bert-base-uncased`, loaded by Hugging Face |
| Visual encoder | EVA-CLIP weights downloaded by the LAVIS loader to Torch Hub's cache |
| BLIP-2 pretraining | LAVIS URL in `lavis_1019/configs/models/blip2/blip2_pretrain_flant5xl.yaml` |
| SMART fine-tuning | Your `checkpoint_best.pth`; default evaluation path is `checkpoints/smart_qvhighlights.pth` |

For existing local model copies, use paths relative to the repository root:

```bash
export SMART_BERT_MODEL=./checkpoints/bert-base-uncased
bash scripts/eval_qvhighlights.sh --options \
  model.t5_model=./checkpoints/flan-t5-xl \
  model.pretrained=checkpoints/blip2_pretrained_flant5xl.pth \
  model.beats_checkpoint=checkpoints/BEATs_iter3.pt \
  model.finetuned=checkpoints/smart_qvhighlights.pth
```

Evaluation loads the pretrained backbone and then the fine-tuned SMART state, so
both are required. `model.pretrained` does not replace the separate EVA-CLIP
initialization; offline use also requires its weights in the Torch Hub cache.

There is no hosted SMART fine-tuned checkpoint in this release. Training produces
the checkpoint used by the evaluation command. See the root README for output
paths and training settings.
