"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import json
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5TokenizerFast
from peft import LoraConfig, get_peft_model
import wandb
from lavis_1019.common.registry import registry
from lavis_1019.common.utils import is_url
from lavis_1019.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis_1019.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis_1019.models.BEATs_model.BEATs import BEATs, BEATsConfig
from lavis_1019.common.dist_utils import is_main_process
from lavis_1019.common.dist_utils import download_cached_file
from lavis_1019.models.blip2_mr_models.utils import (
    format_wandb_log_images_and_predictions,
    post_process,
    post_process_TAL,
    get_timestamps_as_seconds_integers,
    get_timestamps_as_relative_integers,
    get_timestamps_as_seconds_floats,
    get_timestamps_as_relative_floats,
    get_timestamps_as_framenumbers,
)
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@registry.register_model("blip2_mr")
class BLIP2_MR(Blip2Base):
    """SMART moment retrieval with BLIP-2, Flan-T5-XL, BEATs and STC."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        num_beams=5,
        prompt="",
        max_txt_len=200,
        apply_lemmatizer=False,
        input_time_format="seconds_integers",
        interleave_data=False,
        frame_token_aggregation=None,
        stc_ablation_mode="full_stc",
        keyframe_topk=32,
        use_audio=True,
        beats_checkpoint="checkpoints/BEATs_iter3.pt",
        t5_gradient_checkpointing=False,
        stc_debug_dump=False,
        stc_debug_dir=None,
        stc_debug_qids=None,
        task="lora",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        num_query_token = 32

        self.topk = keyframe_topk
        self.task = task
        self.stc_ablation_mode = stc_ablation_mode
        self.use_keyframe_selection = stc_ablation_mode in ["keyframe_only", "full_stc"]
        self.use_intra_shot_token_filter = stc_ablation_mode in ["token_filter_only", "full_stc"]
        valid_stc_modes = ["baseline", "keyframe_only", "token_filter_only", "full_stc"]
        if self.stc_ablation_mode not in valid_stc_modes:
            raise ValueError(
                f"Unsupported stc_ablation_mode: {self.stc_ablation_mode}. "
                f"Expected one of {valid_stc_modes}."
            )
        if "TAL" in task:
            self.post_process = post_process_TAL
        else:
            self.post_process = post_process
        self.use_lora = True if "lora" in task else False
        self.use_wandb = False

        self.log_samples_every_n = 3000
        self.log_samples_every_n_eval = 1000

        self.input_time_format = input_time_format
        self.interleave_data = interleave_data
        self.frame_token_aggregation = frame_token_aggregation
        self.use_audio = use_audio
        self.t5_gradient_checkpointing = t5_gradient_checkpointing
        self.stc_debug_dump = stc_debug_dump
        self.stc_debug_dir = stc_debug_dir
        self.stc_debug_qids = set(str(qid) for qid in (stc_debug_qids or []))
        logging.info(
            "STC ablation mode: %s | use_keyframe_selection=%s | "
            "use_intra_shot_token_filter=%s | use_audio=%s | keyframe_topk=%s | "
            "t5_gradient_checkpointing=%s | stc_debug_dump=%s | stc_debug_qids=%s",
            self.stc_ablation_mode,
            self.use_keyframe_selection,
            self.use_intra_shot_token_filter,
            self.use_audio,
            self.topk,
            self.t5_gradient_checkpointing,
            self.stc_debug_dump,
            sorted(self.stc_debug_qids),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mm_hidden_size = 1408

        if self.use_wandb and is_main_process():
            self.wandb_table_data = []
            self.wandb_table_data_eval = []

        (
            self.visual_encoder,
            self.ln_vision,
        ) = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        if self.t5_gradient_checkpointing:

            self.t5_model.gradient_checkpointing_enable()
            logging.info("Enabled T5 gradient checkpointing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BEATs_model = None
        if self.use_audio:
            if not os.path.isabs(beats_checkpoint):
                beats_checkpoint = os.path.join(registry.get_path("repo_root"), beats_checkpoint)
            if not os.path.isfile(beats_checkpoint):
                raise FileNotFoundError(
                    f"BEATs checkpoint not found: {beats_checkpoint}. "
                    "Download BEATs_iter3.pt or set model.beats_checkpoint in your config."
                )
            BEATs_model = torch.load(beats_checkpoint, map_location=device)  
            BEATs_cfg = BEATsConfig(BEATs_model['cfg'])
            self.BEATs_model = BEATs(BEATs_cfg)
            self.BEATs_model.load_state_dict(BEATs_model['model'])

            self.BEATs_model.to(device)  
            self.BEATs_model.eval()
        else:
            logging.info("Audio branch disabled for this run")

        self.annoying_numbers, _ = self.find_annoying_numbers(self.t5_tokenizer, 200)
        self.annoying_numbers_replacement_dict = (
            self.find_annoying_numbers_replacement_dict(self.annoying_numbers)
        )

        logging.info(
            "Annoying numbers and their replacement: {}".format(
                self.annoying_numbers_replacement_dict
            )
        )

        if self.use_lora:

            model_modules = str(self.t5_model.modules)
            pattern = r"\((\w+)\): Linear"
            linear_layer_names = re.findall(pattern, model_modules)

            names = []

            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))

            lora_config = LoraConfig(
                r=8,
                target_modules=target_modules,
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.t5_model = get_peft_model(self.t5_model, lora_config)
            self.t5_model.print_trainable_parameters()
        else:

            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()

        self.multimodal_Qformer = False

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not self.multimodal_Qformer:
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer_tokenizer = self.init_tokenizer()
        self.Qformer_tokenizer = self.init_tokenizer()
        self.num_query_token = num_query_token
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )
        self.audio_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )
        # Retain these legacy projection parameters for checkpoint and optimizer compatibility.
        self.vis_proj = nn.Linear(
            self.mm_hidden_size, self.t5_model.config.hidden_size
        )

        self.visual_t5_projector = nn.Linear(self.mm_hidden_size, self.t5_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.num_beams = num_beams

        self.seperator_token = self.t5_tokenizer.convert_tokens_to_ids(">")
        self.pad_token_id = self.t5_tokenizer.pad_token_id

        if "qformer_freeze" in self.task:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            # Preserve the original training behavior: this sets a module attribute.
            self.t5_proj.requires_grad = False

        if is_main_process() and self.use_wandb:
            if self.use_lora:
                wandb.watch(self.t5_model, log="all")
            if not "qformer_freeze" in self.task:
                wandb.watch(self.Qformer, log="all")
                wandb.watch(self.t5_proj, log="all")

    def compress_qformer(self, frames):
        """Merge the most similar adjacent tokens until 24 remain."""
        n_frames, emb_dim = frames.shape
        while n_frames > 24:

            similarities = F.cosine_similarity(frames[:-1, :], frames[1:, :], dim=-1)

            max_sim_index = torch.argmax(similarities)

            merged_frame = (frames[max_sim_index] + frames[max_sim_index + 1]) / 2

            frames = torch.cat((frames[:max_sim_index], merged_frame.unsqueeze(0), frames[max_sim_index + 2:]), dim=0)

            n_frames = frames.shape[0]
        
        return frames

    def compress_qformer_std(self, frame_emb, topstd_list):

        compressed_frame_emb = frame_emb[topstd_list, :]
        return compressed_frame_emb

    def selected_qformer(self, frames_emb, t, return_scores=False):
        """Select 16 token positions by their temporal variation within a shot."""

        query_num, emb_dim = frames_emb.shape[0] // t, frames_emb.shape[1]

        frames_emb = frames_emb.view(t, query_num, emb_dim)

        changes_across_t = torch.zeros((query_num))
        frames_for_single_batch = frames_emb.permute(1, 0, 2)

        variation_scores = []
        for j in range(query_num):
            temporal_feature = frames_for_single_batch[j]

            std_dev = temporal_feature.std(dim=0).mean()
            variation_scores.append(std_dev)
        changes_across_t = torch.tensor(variation_scores)

        top16_indices = torch.topk(changes_across_t.float(), k=16, dim=0).indices.sort().values.tolist()
        if return_scores:
            return top16_indices, changes_across_t
        return top16_indices

    def compress_audio_features(self, audio_feat, target_tokens=150, method='avg_pooling'):
        """Pool audio features from [batch, tokens, dim] to a fixed token count."""
        b, n_tokens, dim = audio_feat.shape

        if method == 'avg_pooling':

            pooled_feats = F.adaptive_avg_pool1d(audio_feat.transpose(1, 2), target_tokens).transpose(1, 2)

        elif method == 'max_pooling':

            pooled_feats = F.adaptive_max_pool1d(audio_feat.transpose(1, 2), target_tokens).transpose(1, 2)

        elif method == 'linear':

            linear_layer = torch.nn.Linear(n_tokens, target_tokens)
            pooled_feats = linear_layer(audio_feat.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError("Invalid method. Choose from 'avg_pooling', 'max_pooling', or 'linear'.")
        return pooled_feats

    def extract_audio_features(self, audio):
        """Encode audio into projected BEATs features when the audio branch is enabled."""
        if not self.use_audio:
            return None

        audio = audio.mean(dim=-1)
        chunk_size = 16000 * 10
        chunks = audio.split(chunk_size, dim=1)
        audio_feats = []
        for chunk in chunks:
            if chunk.size(1) < chunk_size:
                padding = torch.zeros((1, chunk_size - chunk.size(1)), device=chunk.device)
                chunk = torch.cat((chunk, padding), dim=1)
            chunk = chunk.to("cuda")
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    audio_feats.append(self.BEATs_model.extract_features(chunk)[0])

        audio_feat = torch.cat(audio_feats, dim=1)
        audio_feat = self.compress_audio_features(audio_feat)
        audio_feat = self.audio_proj(audio_feat)
        return audio_feat

    def process_scenes(self, image_embeds, scene_labels, frames_per_scene):
        t = len(scene_labels)  
        assert image_embeds.shape[0] == t * frames_per_scene, "image_embeds must have t * frames_per_scene rows"
        scene_counts = defaultdict(int)
        scene_indices = defaultdict(list) 
        for idx, label in enumerate(scene_labels.tolist()):
            scene_counts[label] += 1
            scene_indices[label].append(idx)  
        scene_results = {}  
        for label, indices in scene_indices.items():
            count = scene_counts[label]
            image_indices = torch.cat([torch.arange(i * frames_per_scene, (i + 1) * frames_per_scene) for i in indices])
            frames_emb = image_embeds[image_indices]  
            scene_results[label] = {'count': count}
            if count > 1:
                scene_results[label]['qformer_output'] = self.selected_qformer(frames_emb, count)
        return scene_results

    def build_keyframe_labels(self, image_embeds_tmp):
        """
        Build frame-level keyframe labels once and share the same logic between
        training and generation for STC ablations.
        """
        labels = []
        for single_image_embeds in image_embeds_tmp:
            diff_embeds = torch.zeros_like(single_image_embeds)
            diff_embeds[1:, :, :] = single_image_embeds[1:, :, :] - single_image_embeds[:-1, :, :]
            change_magnitude = torch.norm(diff_embeds, dim=(1, 2))
            label = torch.zeros_like(change_magnitude, dtype=torch.long)

            if self.use_keyframe_selection:
                _, topk_indices = torch.topk(change_magnitude, self.topk, dim=0)
                label[topk_indices] = 1
            else:

                label[:] = 1

            labels.append(label)

        return labels

    def apply_intra_shot_token_filter(self, frame_emb, scene_info):
        """Apply the shot-aware token filtering branch used by STC."""
        if scene_info["count"] <= 1:
            return self.compress_qformer(frame_emb)

        return self.compress_qformer_std(frame_emb, scene_info["qformer_output"])

    def apply_stc_ablation_to_frame(self, frame_emb, label_value, scene_info, frame_index):
        """
        Map each ablation setting to the corresponding frame-token behavior.

        baseline:
            No STC at all.
        keyframe_only:
            Keep keyframes untouched and use the generic token compressor on
            non-keyframes. This isolates the frame-level keyframe selection stage.
        token_filter_only:
            Apply intra-shot token filtering to every frame without keyframe gating.
        full_stc:
            Keep keyframes untouched and apply intra-shot token filtering only to
            non-keyframes, matching the current SMART implementation.
        """
        if self.stc_ablation_mode == "baseline":
            return frame_emb

        if self.stc_ablation_mode == "keyframe_only":
            if label_value == 0 and frame_index != 0:
                return self.compress_qformer(frame_emb)
            return frame_emb

        if self.stc_ablation_mode == "token_filter_only":
            return self.apply_intra_shot_token_filter(frame_emb, scene_info)

        if self.stc_ablation_mode == "full_stc":
            if label_value == 0 and frame_index != 0:
                return self.apply_intra_shot_token_filter(frame_emb, scene_info)
            return frame_emb

        raise ValueError(f"Unhandled stc_ablation_mode: {self.stc_ablation_mode}")

    def build_visual_token_stats(self, tokens_before, tokens_after):
        """Build a serializable summary for visual token retention."""
        token_kept_pct = []
        for before, after in zip(tokens_before, tokens_after):
            if before == 0:
                token_kept_pct.append(0.0)
            else:
                token_kept_pct.append((after / before) * 100.0)

        return {
            "visual_tokens_before": tokens_before,
            "visual_tokens_after": tokens_after,
            "visual_tokens_kept_pct": token_kept_pct,
        }

    def should_dump_stc_debug(self, qid):
        if torch.is_tensor(qid):
            qid = qid.item()
        if not self.stc_debug_dump:
            return False
        if not self.stc_debug_qids:
            return is_main_process()
        return str(qid) in self.stc_debug_qids

    def compute_patch_temporal_change_maps(self, single_image_embeds):
        token_count = single_image_embeds.shape[1]
        patch_offset = 1 if int(math.isqrt(max(token_count - 1, 1))) ** 2 == token_count - 1 else 0
        patch_tokens = single_image_embeds[:, patch_offset:, :]
        patch_count = patch_tokens.shape[1]
        grid_size = int(math.isqrt(max(patch_count, 1)))
        if grid_size * grid_size != patch_count:
            return None

        diff_tokens = torch.zeros_like(patch_tokens)
        diff_tokens[1:, :, :] = patch_tokens[1:, :, :] - patch_tokens[:-1, :, :]
        patch_scores = torch.norm(diff_tokens, dim=-1)
        return patch_scores.view(single_image_embeds.shape[0], grid_size, grid_size)

    def compute_shot_token_variance(self, shot_frame_tokens):
        if shot_frame_tokens.ndim != 3:
            raise ValueError(
                f"Expected shot_frame_tokens to have shape [num_frames, num_tokens, dim], got {shot_frame_tokens.shape}"
            )

        token_variance_scores = shot_frame_tokens.permute(1, 0, 2).std(dim=1).mean(dim=1)
        topk = min(16, token_variance_scores.shape[0])
        selected_token_indices = (
            torch.topk(token_variance_scores.float(), k=topk, dim=0).indices.sort().values
        )
        return token_variance_scores, selected_token_indices

    def denormalize_video_frames(self, video_tensor):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=video_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=video_tensor.device).view(1, 3, 1, 1)
        frames = video_tensor * std + mean
        frames = frames.clamp(0.0, 1.0)
        frames = (frames * 255.0).round().to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()

    def dump_stc_debug(self, samples, image_embeds_tmp, labels, frames_for_t5_by_frame):
        if not self.stc_debug_dump:
            return

        debug_dir = self.stc_debug_dir or os.path.join(os.getcwd(), "logs", "stc_debug")
        os.makedirs(debug_dir, exist_ok=True)

        qids = samples["query_id"]
        video_names = samples["video_name"]
        query_texts = samples["query_text"]
        frame_indices_batch = samples["frame_indices"]
        timestamps_batch = samples["timestamps"]
        scene_labels_batch = samples["scene_labels"]
        scene_start_frames_batch = samples["scene_start_frames_debug"]
        scene_start_times_batch = samples["scene_start_times_debug"]
        videos = samples["video"]

        for batch_idx, qid in enumerate(qids):
            if torch.is_tensor(qid):
                qid = qid.item()
            if not self.should_dump_stc_debug(qid):
                continue

            single_image_embeds = image_embeds_tmp[batch_idx].detach().cpu()
            change_magnitude = torch.zeros(single_image_embeds.shape[0], dtype=torch.float32)
            change_magnitude[1:] = torch.norm(
                single_image_embeds[1:, :, :] - single_image_embeds[:-1, :, :], dim=(1, 2)
            )
            patch_change_maps = self.compute_patch_temporal_change_maps(single_image_embeds)
            label_tensor = labels[batch_idx].detach().cpu()
            scene_label_tensor = scene_labels_batch[batch_idx].detach().cpu()
            frame_indices = frame_indices_batch[batch_idx].detach().cpu()
            timestamps = timestamps_batch[batch_idx].detach().cpu()
            frame_tokens = frames_for_t5_by_frame[batch_idx].detach().cpu()

            shots = []
            unique_shot_ids = []
            for shot_id in scene_label_tensor.tolist():
                if shot_id not in unique_shot_ids:
                    unique_shot_ids.append(shot_id)

            for shot_id in unique_shot_ids:
                frame_positions = (scene_label_tensor == shot_id).nonzero(as_tuple=False).flatten()
                if frame_positions.numel() == 0:
                    continue
                shot_frame_tokens = frame_tokens[frame_positions]
                token_variance_scores, selected_token_indices = self.compute_shot_token_variance(
                    shot_frame_tokens
                )
                shots.append(
                    {
                        "shot_id": int(shot_id),
                        "frame_positions_in_sample": frame_positions.tolist(),
                        "sampled_frame_indices": frame_indices[frame_positions].tolist(),
                        "sampled_timestamps": timestamps[frame_positions].tolist(),
                        "token_variance_scores": token_variance_scores.tolist(),
                        "selected_token_indices": selected_token_indices.tolist(),
                    }
                )

            metadata = {
                "query_id": int(qid) if str(qid).isdigit() else str(qid),
                "video_name": video_names[batch_idx],
                "query_text": query_texts[batch_idx],
                "n_frames": int(frame_indices.shape[0]),
                "frame_indices": frame_indices.tolist(),
                "timestamps": timestamps.tolist(),
                "scene_labels": scene_label_tensor.tolist(),
                "scene_start_frames": json.loads(scene_start_frames_batch[batch_idx]),
                "scene_start_times": json.loads(scene_start_times_batch[batch_idx]),
                "frame_scores": change_magnitude.tolist(),
                "keyframe_mask": label_tensor.tolist(),
                "selected_keyframes": (label_tensor == 1).nonzero(as_tuple=False).flatten().tolist(),
                "shots": shots,
                "visualization_versions": {
                    "v1": "frame dynamic score + keyframe markers + token variance bar plot",
                    "v2": "frame dynamic score + keyframe markers + patch temporal-change heatmap",
                },
            }

            stem = f"qid_{metadata['query_id']}"
            json_path = os.path.join(debug_dir, f"{stem}.json")
            npz_path = os.path.join(debug_dir, f"{stem}.npz")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            npz_payload = {
                "frames_uint8": self.denormalize_video_frames(videos[batch_idx].detach()),
            }
            if patch_change_maps is not None:
                npz_payload["patch_change_maps"] = patch_change_maps.cpu().numpy().astype(np.float32)
            np.savez_compressed(npz_path, **npz_payload)
            logging.info("Dumped STC debug sample for qid=%s to %s", qid, json_path)

    def forward(
        self,
        samples,
    ):
        image = samples["video"]
        scene_labels = samples["scene_labels"]
        timestamps, durations = (
            samples["timestamps"],
            samples["duration"],
        )
        
        time_begin = samples["time_begin_token"]
        time_end = samples["time_end_token"]
        frame_begin = samples["frame_begin_token"]
        frame_end = samples["frame_end_token"]
        video_prompt_end = samples["video_prompt_end"]
        audio_prompt_end = samples["audio_prompt_end"]
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        audio = samples["audio"]
        answer = samples["relevant_windows"]
        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))

        visual_last_two_dims = image_embeds.shape[-2:]
        image_embeds_tmp = image_embeds.view(b, t, *visual_last_two_dims)

        labels = self.build_keyframe_labels(image_embeds_tmp)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.multimodal_Qformer:
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text = self.Qformer_tokenizer(
                [
                    q for q in query_prompt for _ in range(t)
                ],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            qformer_output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            frames_for_projection = qformer_output.last_hidden_state[
                :, : query_tokens.size(1), :
            ]
        else:
            frames_after_qformer = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            frames_for_projection = frames_after_qformer.last_hidden_state

        frames_for_t5 = self.t5_proj(frames_for_projection)
        frames_for_t5 = frames_for_t5.reshape(
            b, t, frames_for_t5.shape[-2], -1
        )
        frames_atts_for_t5 = torch.ones(frames_for_t5.size()[:-1], dtype=torch.long).to(
            image.device
        )
        frames_for_t5 = frames_for_t5.reshape(
            b, -1, frames_for_t5.shape[-1]
        )
        frames_atts_for_t5 = frames_atts_for_t5.reshape(b, -1)

        audio_feat = self.extract_audio_features(audio)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt, token_stats = self.prompt_concatenation(
                timestamps,
                durations,
                frames_for_t5,
                frames_atts_for_t5,
                audio_feat,
                video_prompt_end,
                audio_prompt_end,
                time_begin,
                time_end,
                frame_begin,
                frame_end,
                query_prompt,
                task_prompt,
                labels,
                scene_labels
            )

            output_tokens_mr = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            targets_mr = output_tokens_mr.input_ids.masked_fill(
                output_tokens_mr.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            output_tokens_mask_mr = output_tokens_mr.attention_mask

            outputs_loc = self.t5_model(
                inputs_embeds=inputs_embs_mr,
                attention_mask=inputs_atts_mr,
                decoder_attention_mask=output_tokens_mask_mr,
                return_dict=True,
                labels=targets_mr,
            )

            loss = outputs_loc.loss
            pred = self.t5_tokenizer.batch_decode(
                torch.argmax(outputs_loc.logits, dim=-1)
            )

            if self.use_wandb and is_main_process():
                log = {}
                log["train/log_likelihood_loss"] = loss.item()

                if samples["iters"] % self.log_samples_every_n == 0:
                    pred = self.t5_tokenizer.batch_decode(
                        torch.argmax(outputs_loc.logits, dim=-1)
                    )

                    out, self.wandb_table_data = (
                        format_wandb_log_images_and_predictions(
                            samples=samples,
                            wandb_table_data=self.wandb_table_data,
                            pred=pred,
                            video_prompt=video_prompt,
                            post_process_fn=self.post_process,
                            input_time_format=self.input_time_format,
                            interleave_data=True,
                            train_data=True,
                        )
                    )
                    log.update(out)

                wandb.log(log)

        return {
            "loss": loss,
            "token_stats": token_stats,
        }

    def prompt_concatenation(
        self,
        timestamps,
        durations,
        frames_for_t5,
        frames_atts_for_t5,
        audio_feat,
        video_prompt_end,
        audio_prompt_end,
        time_begin,
        time_end,
        frame_begin,
        frame_end,
        query_prompt,
        task_prompt,
        labels,
        scene_labels
    ):

        time_begin_token = self.t5_tokenizer(
            time_begin[0],
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        time_begin_emb = self.t5_model.encoder.embed_tokens(
            time_begin_token.input_ids
        ).squeeze(1)
        time_end_token = self.t5_tokenizer(
            time_end[0],
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        time_end_emb = self.t5_model.encoder.embed_tokens(
            time_end_token.input_ids
        ).squeeze(1)
        frame_begin_token = self.t5_tokenizer(
            frame_begin[0],
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        frame_begin_emb = self.t5_model.encoder.embed_tokens(
            frame_begin_token.input_ids
        ).squeeze(1)
        frame_end_token = self.t5_tokenizer(
            frame_end[0],
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        frame_end_emb = self.t5_model.encoder.embed_tokens(
            frame_end_token.input_ids
        ).squeeze(1)

        if "only_frames" in self.task:
            assert (
                not self.input_time_format
            ), "Set input_time_format to False in the config to use only frames without timestamps."

            video_prompt = ["<vid>" for _ in range(len(timestamps))]
            video_prompt_end = ["<extra_id_0>\n" for _ in range(len(video_prompt_end))]
        elif "add_duration" in self.task:

            video_prompt_end = [
                "{}<extra_id_0>\n".format(">" + d.item()) for d in durations
            ]
            video_prompt = ["<vid>" for _ in range(len(timestamps))]

        if self.input_time_format == "framenumbers":
            timestamps, durations, video_prompt = get_timestamps_as_framenumbers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "relative_floats":
            timestamps, durations, video_prompt = get_timestamps_as_relative_floats(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "relative_integers":
            timestamps, durations, video_prompt = get_timestamps_as_relative_integers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "seconds_integers":
            timestamps, durations, video_prompt = get_timestamps_as_seconds_integers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "seconds_floats":
            timestamps, durations, video_prompt = get_timestamps_as_seconds_floats(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        video_prompt_end_tokens = self.t5_tokenizer(
            video_prompt_end,
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)

        video_prompt_end_embs = self.t5_model.encoder.embed_tokens(
            video_prompt_end_tokens.input_ids
        )
        if self.use_audio:
            audio_prompt_end_tokens = self.t5_tokenizer(
                audio_prompt_end,
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(frames_for_t5.device)
            audio_prompt_end_embs = self.t5_model.encoder.embed_tokens(
                audio_prompt_end_tokens.input_ids
            )

        if "no_task_prompt" in self.task:
            text_prompt = [q for q in query_prompt]
        else:
            text_prompt = [q + t for q, t in zip(query_prompt, task_prompt)]
        text_prompt_tokens = self.t5_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)

        text_prompt_embs = self.t5_model.encoder.embed_tokens(
            text_prompt_tokens.input_ids
        )
        if self.interleave_data:
            assert (
                "integer" in self.input_time_format
            ), "Interleaving only works with integer time formats where one number is one token."

            batch_timestamps_tokens = []
            batch_timestamps_embs = []
            for t in timestamps:
                tokens, embs = self.get_clean_timestamp_tokens_and_embs(t) 
                batch_timestamps_tokens.append(tokens)
                batch_timestamps_embs.append(embs)
            
            batch_duration_tokens, batch_duration_embs = (
                self.get_clean_timestamp_tokens_and_embs(torch.tensor(durations))
            )
            interleaved_video_prompt_embs = []
            video_prompt = []
            visual_tokens_before = []
            visual_tokens_after = []

            b, t_n, c = frames_for_t5.shape

            if self.frame_token_aggregation:
                n = 1
            else:
                n = int(self.num_query_token)
            t = t_n // n

            for j, (timestamp_embs, frame_embs) in enumerate(
                zip(batch_timestamps_embs, frames_for_t5)
            ):
                scene_label = scene_labels[j]
                scene_results = self.process_scenes(frame_embs, scene_label, n)
                label = labels[j]
                interleaved_prompt = torch.tensor([]).to(frames_for_t5.device)
                sample_visual_tokens_before = 0
                sample_visual_tokens_after = 0

                _video_prompt = []

                for i in range(t):

                    frame_emb = frame_embs[i * n : i * n + n]
                    sample_visual_tokens_before += frame_emb.shape[0]

                    timestamp_emb = timestamp_embs[i]

                    _video_prompt.append(
                        "f{i}-".format(i=i)
                        + self.t5_tokenizer.decode(batch_timestamps_tokens[j][i])
                        + ">"
                    )

                    scene_info = scene_results[int(scene_label[i])]

                    frame_emb = self.apply_stc_ablation_to_frame(
                        frame_emb,
                        int(label[i].item()),
                        scene_info,
                        i,
                    )
                    sample_visual_tokens_after += frame_emb.shape[0]
                    frame_and_time = torch.cat(
                        [
                            time_begin_emb,
                            timestamp_emb,
                            time_end_emb,
                            frame_begin_emb,
                            frame_emb,
                            frame_end_emb
                        ]
                    )

                    interleaved_prompt = torch.cat([interleaved_prompt, frame_and_time])

                seperator_emb = self.t5_model.encoder.embed_tokens(
                    torch.tensor(self.seperator_token)
                    .to(frames_for_t5.device)
                    .unsqueeze(0)
                )
                duration_emb = batch_duration_embs[j]
                interleaved_prompt = torch.cat(
                    [interleaved_prompt, seperator_emb, duration_emb]
                )

                interleaved_video_prompt_embs.append(interleaved_prompt)
                visual_tokens_before.append(sample_visual_tokens_before)
                visual_tokens_after.append(sample_visual_tokens_after)

                video_prompt.append(
                    "".join(_video_prompt)
                    + self.t5_tokenizer.decode(batch_duration_tokens[j])
                )

            max_len = max([len(i) for i in interleaved_video_prompt_embs])
            for i in range(len(interleaved_video_prompt_embs)):
                if len(interleaved_video_prompt_embs[i]) < max_len:
                    padding = self.pad_token_id * torch.ones(
                        max_len - len(interleaved_video_prompt_embs[i]),
                        interleaved_video_prompt_embs[i].shape[-1],
                    ).to(frames_for_t5.device)
                    interleaved_video_prompt_embs[i] = torch.cat(
                        [padding, interleaved_video_prompt_embs[i]]
                    )

            interleaved_video_prompt_embs = torch.stack(
                interleaved_video_prompt_embs
            ).to(frames_for_t5.device)
            token_stats = self.build_visual_token_stats(
                visual_tokens_before,
                visual_tokens_after,
            )

            input_emb_segments = [
                interleaved_video_prompt_embs,
                video_prompt_end_embs,
            ]
            if self.use_audio:
                input_emb_segments.extend([audio_feat, audio_prompt_end_embs])
            input_emb_segments.append(text_prompt_embs)
            inputs_embs_mr = torch.cat(input_emb_segments, dim=1)
            interleaved_video_prompt_attn_embs = torch.ones(
                interleaved_video_prompt_embs.size()[:-1], dtype=torch.long
            ).to(frames_for_t5.device)
            interleaved_video_prompt_attn_embs = (
                interleaved_video_prompt_attn_embs.reshape(b, -1)
            )
            attn_segments = [
                interleaved_video_prompt_attn_embs,
                video_prompt_end_tokens.attention_mask,
            ]
            if self.use_audio:
                audio_feat_attn_embs = torch.ones(
                    audio_feat.size()[:-1], dtype=torch.long
                ).to(frames_for_t5.device)
                audio_feat_atts = audio_feat_attn_embs.reshape(b, -1)
                attn_segments.extend(
                    [audio_feat_atts, audio_prompt_end_tokens.attention_mask]
                )
            attn_segments.append(text_prompt_tokens.attention_mask)
            inputs_atts_mr = torch.cat(attn_segments, dim=1)
        else:

            visual_tokens_before = [int(frames_for_t5.shape[1]) for _ in range(frames_for_t5.shape[0])]
            visual_tokens_after = [int(frames_for_t5.shape[1]) for _ in range(frames_for_t5.shape[0])]
            token_stats = self.build_visual_token_stats(
                visual_tokens_before,
                visual_tokens_after,
            )

            video_prompt_tokens = self.t5_tokenizer(
                video_prompt,
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(frames_for_t5.device)
            video_prompt_embs = self.t5_model.encoder.embed_tokens(
                video_prompt_tokens.input_ids
            )

            inputs_embs_mr = torch.cat(
                [
                    video_prompt_embs,
                    frames_for_t5,
                    video_prompt_end_embs,
                    text_prompt_embs,
                ],
                dim=1,
            )

            inputs_atts_mr = torch.cat(
                [
                    video_prompt_tokens.attention_mask,
                    frames_atts_for_t5,
                    video_prompt_end_tokens.attention_mask,
                    text_prompt_tokens.attention_mask,
                ],
                dim=1,
            )

            video_prompt = [
                p + "frames" + end_p
                for (p, end_p) in zip(video_prompt, video_prompt_end)
            ]

        return inputs_embs_mr, inputs_atts_mr, video_prompt, token_stats

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=8,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image = samples["video"]
        scene_labels = samples["scene_labels"]
        timestamps, durations = (
            samples["timestamps"],
            samples["duration"],
        )
        time_begin = samples["time_begin_token"]
        time_end = samples["time_end_token"]
        frame_begin = samples["frame_begin_token"]
        frame_end = samples["frame_end_token"]
        qid = samples["query_id"]
        video_prompt_end = samples["video_prompt_end"]
        audio_prompt_end = samples["audio_prompt_end"]
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        answer = samples["relevant_windows"]
        audio = samples["audio"]

        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        _, n, _ = image_embeds.shape

        visual_last_two_dims = image_embeds.shape[-2:]
        image_embeds_tmp = image_embeds.view(b, t, *visual_last_two_dims)

        labels = self.build_keyframe_labels(image_embeds_tmp)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.multimodal_Qformer:
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text = self.Qformer_tokenizer(
                [
                    q for q in query_prompt for _ in range(t)
                ],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            qformer_output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            frames_for_projection = qformer_output.last_hidden_state[
                :, : query_tokens.size(1), :
            ]
        else:
            
            frames_after_qformer = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            frames_for_projection = frames_after_qformer.last_hidden_state
        frames_for_t5 = self.t5_proj(frames_for_projection)

        if self.frame_token_aggregation:
            assert self.frame_token_aggregation in [
                "mean"
            ], "Invalid aggregation method, please choose from ['mean']"
            frames_for_t5 = frames_for_t5.mean(dim=1, keepdim=True)

        frames_for_t5 = frames_for_t5.reshape(
            b, t, frames_for_t5.shape[-2], -1
        )
        self.dump_stc_debug(samples, image_embeds_tmp, labels, frames_for_t5)
        frames_atts_for_t5 = torch.ones(frames_for_t5.size()[:-1], dtype=torch.long).to(
            image.device
        )
        frames_for_t5 = frames_for_t5.reshape(
            b, -1, frames_for_t5.shape[-1]
        )
        frames_atts_for_t5 = frames_atts_for_t5.reshape(b, -1)
        
        audio_feat = self.extract_audio_features(audio)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt, token_stats = self.prompt_concatenation(
                timestamps,
                durations,
                frames_for_t5,
                frames_atts_for_t5,
                audio_feat,
                video_prompt_end,
                audio_prompt_end,
                time_begin,
                time_end,
                frame_begin,
                frame_end,
                query_prompt,
                task_prompt,
                labels,
                scene_labels
            )

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embs_mr,
                attention_mask=inputs_atts_mr,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=output_attentions,
            )

            pred_ans = self.t5_tokenizer.batch_decode(
                outputs[0], skip_special_tokens=True
            )

        if isinstance(samples["duration"], torch.Tensor):
            out["duration"] = samples["duration"].tolist()
        else:
            out["duration"] = samples["duration"]

        if (
            self.input_time_format == "relative_integers"
            or self.input_time_format == "relative_floats"
        ):
            prediction = [self.post_process(pred) for pred in pred_ans]
            out["prediction"] = self.convert_to_absolute_time(
                prediction, out["duration"]
            )
        else:

            out["prediction"] = [self.post_process(pred) for pred in pred_ans]

        out["raw_prediction"] = pred_ans
        out["answer"] = answer

        out["qid"] = qid
        out["token_stats"] = token_stats

        if self.use_wandb and is_main_process():

            if samples["iters"] % self.log_samples_every_n_eval == 0:
                table, self.wandb_table_data_eval = (
                    format_wandb_log_images_and_predictions(
                        samples=samples,
                        wandb_table_data=self.wandb_table_data_eval,
                        pred=pred_ans,
                        video_prompt=video_prompt,
                        post_process_fn=self.post_process,
                        input_time_format=self.input_time_format,
                        interleave_data=True,
                        train_data=False,
                    )
                )

                wandb.log(table)

        return out

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        num_beams = cfg.get("num_beams", 5)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        input_time_format = cfg.get("input_time_format", "seconds_integers")
        interleave_data = cfg.get("interleave_data", True)
        frame_token_aggregation = cfg.get("frame_token_aggregation", None)
        stc_ablation_mode = cfg.get("stc_ablation_mode", "full_stc")
        keyframe_topk = cfg.get("keyframe_topk", 32)
        use_audio = cfg.get("use_audio", True)
        t5_gradient_checkpointing = cfg.get("t5_gradient_checkpointing", False)
        stc_debug_dump = cfg.get("stc_debug_dump", False)
        stc_debug_dir = cfg.get("stc_debug_dir", None)
        stc_debug_qids = cfg.get("stc_debug_qids", [])

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_len", 200)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", "qformer_freeze_lora")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            num_beams=num_beams,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            input_time_format=input_time_format,
            interleave_data=interleave_data,
            frame_token_aggregation=frame_token_aggregation,
            stc_ablation_mode=stc_ablation_mode,
            keyframe_topk=keyframe_topk,
            use_audio=use_audio,
            beats_checkpoint=cfg.get("beats_checkpoint", "checkpoints/BEATs_iter3.pt"),
            t5_gradient_checkpointing=t5_gradient_checkpointing,
            stc_debug_dump=stc_debug_dump,
            stc_debug_dir=stc_debug_dir,
            stc_debug_qids=stc_debug_qids,
            task=task,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:

            pretrain_path = cfg.get("pretrained", None)

            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info("load finetuned weights from %s" % finetune_path)
        else:

            pretrain_path = cfg.get("pretrained", None)

            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

    def load_from_pretrained(self, url_or_filename, **kwargs):
        """
        Load BLIP2 pretraining weights.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

    def find_annoying_numbers(
        self,
        tokenizer=None,

        range_end=300,
    ):
        """
        Find numbers that are tokenized in more than one token by the T5 tokenizer.

        Args:
            tokenizer: A tokenizer object from the transformers library.
            range_end: The range of numbers to check.

        Returns:
            annoying_numbers: A list of numbers that are tokenized in more than one token.
            annoying_numbers_spance: A list of numbers that are tokenized in more than one token, but the first token is a space.
        """

        if tokenizer is None:
            tokenizer = self.t5_tokenizer

        annoying_numbers = []
        annoying_numbers_spance = []
        for i in range(0, range_end):
            tokens = tokenizer(
                str(i),
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=300,
                return_tensors="pt",
            )

            n_tokens = len(tokens["input_ids"].tolist()[0])

            if n_tokens > 1:
                if tokens["input_ids"].tolist()[0][0] == 3:
                    annoying_numbers_spance.append(i)
                else:
                    annoying_numbers.append(i)

        return annoying_numbers, annoying_numbers_spance

    def find_annoying_numbers_replacement_dict(self, annoying_numbers):
        """
        Find a the closes integer replacement for numbers that are tokenized in more than one token by the T5 tokenizer.

        Args:
            annoying_numbers: A list of numbers that are tokenized in more than one token.

        Returns:
            annoying_numbers_replacement_dict: A dictionary with the number as key and the replacement as value.
        """

        annoying_numbers_replacement_dict = {}
        for i in annoying_numbers:
            for j in range(100):
                if (i + j) not in annoying_numbers:
                    new_i = i + j
                    break
                if (i - j) not in annoying_numbers:
                    new_i = i - j
                    break

            annoying_numbers_replacement_dict[i] = new_i

        return annoying_numbers_replacement_dict

    def get_clean_timestamp_tokens_and_embs(self, timestamps):
        """
        Tokenize timestamps and clean up removing the special tokens.
        Return a list of tokenized and embedded timestamps.
        Each timestamp can be one or more tokens.

        Args:
            timestamps: A list of timestamps.

        Returns:
            tokens: A list of cleaned tokenized timestamps.
            token_embs list(torch.tensor()): A list of token embeddings.
        """

        tokens = self.t5_tokenizer(
            [str(t.item()) for t in timestamps], add_special_tokens=False
        )["input_ids"]

        tokens = [token[1:] if token[0] == 3 else token for token in tokens]

        concatenated_tokens = []
        concatenated_tokens_mask = []
        for i, token in enumerate(tokens):
            concatenated_tokens.extend(token)
            concatenated_tokens_mask.extend([i] * len(token))

        token_embs = []

        token_embs_tmp = self.t5_model.encoder.embed_tokens(
            torch.tensor(concatenated_tokens).to(self.device)
        )

        token_embs = [[] for _ in range(len(timestamps))]
        for i, mask in enumerate(concatenated_tokens_mask):
            if token_embs[mask] == []:
                token_embs[mask] = token_embs_tmp[i].unsqueeze(0)
            else:
                token_embs[mask] = torch.cat(
                    [token_embs[mask], token_embs_tmp[i].unsqueeze(0)],
                    dim=0,
                )

        return tokens, token_embs

