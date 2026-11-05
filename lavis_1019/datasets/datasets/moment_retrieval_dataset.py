import json
import os
import hashlib

import torch
from lavis_1019.datasets.datasets.base_dataset import BaseDataset
from lavis_1019.common.utils import get_cache_path
from PIL import Image
import cv2
from decord import VideoReader
from lavis_1019.processors import transforms_video
from scenedetect import detect, ContentDetector
from scenedetect.scene_manager import FrameTimecode
from moviepy.editor import VideoFileClip
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import numpy as np
import bisect

class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__

class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__

def reduce_frames_manhattan(frms, all_frame_indices, all_scene_labels, target_length=60):
    while frms.shape[1] > target_length:
        n_frms = frms.shape[1]
        diffs = torch.abs(frms[:, 1:, :, :] - frms[:, :-1, :, :]).sum(dim=(0, 1, 2, 3))
        min_diff_idx = torch.argmin(diffs).item()
        frms = torch.cat((frms[:, :min_diff_idx, :, :], frms[:, min_diff_idx+1:, :, :]), dim=1)
        all_frame_indices.pop(min_diff_idx)
        all_scene_labels.pop(min_diff_idx)

    return frms, all_frame_indices, all_scene_labels

def save_frames_to_disk(frms, frame_indices, save_dir, prefix="frame"):
    os.makedirs(save_dir, exist_ok=True) 
    for idx, frame_idx in enumerate(frame_indices):
        frame = frms[:, idx].cpu().numpy().transpose(1, 2, 0) 
        frame_image = Image.fromarray((frame * 255).astype(np.uint8))  
        frame_filename = os.path.join(save_dir, f"{prefix}_{frame_idx}.png")
        frame_image.save(frame_filename)
        print(f"Saved frame {frame_idx} to {frame_filename}")

class MomentRetrievalDataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        audio_shift_mode="aligned",
        audio_shuffle_chunk_seconds=2,
        scene_boundary_mode="default",
        scene_boundary_perturb_frames=5,
        scene_boundary_corruption_ratio=0.2,
        pseudo_shot_seconds=3.0,
        scene_data_file="data/qvhighlights/scene_boundaries.json",
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms_video.NormalizeVideo(mean, std)
        self.image_size = 224
        min_scale = 0.5
        max_scale = 1.0
        self.transform = transforms.Compose(
            [
                transforms_video.RandomResizedCropVideo(
                    self.image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(), 
                ToUint8(),
                transforms_video.ToTensorVideo(),
                self.normalize,
            ]
        )
        self.scene_data = self.load_scene_data(get_cache_path(scene_data_file))
        self.audio_shift_mode = audio_shift_mode
        self.audio_shuffle_chunk_seconds = audio_shuffle_chunk_seconds
        self.scene_boundary_mode = scene_boundary_mode
        self.scene_boundary_perturb_frames = scene_boundary_perturb_frames
        self.scene_boundary_corruption_ratio = scene_boundary_corruption_ratio
        self.pseudo_shot_seconds = pseudo_shot_seconds

    def load_scene_data(self, scene_data_file):
        with open(scene_data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def pad_frames(self, frms, target_frames=80):
        current_frames = frms.shape[1]
        if current_frames < target_frames:
            last_frame = frms[:, -1:, :, :]
            padding_needed = target_frames - current_frames
            frms_padded = torch.cat((frms, last_frame.expand(-1, padding_needed, -1, -1)), dim=1)
        else:
            frms_padded = frms
        return frms_padded

    def extract_audio(self, video_path, start=None, end=None):
        """Extract audio from video within the specified time range and return as a tensor."""
        clip = VideoFileClip(video_path)

        if start is not None and end is not None:
            clip = clip.subclip(start, end)

        audio = clip.audio

        audio_array = audio.to_soundarray(fps=16000)  
        clip.close()  
        return torch.tensor(audio_array, dtype=torch.float32)

    def apply_audio_eval_perturbation(self, audio, qid, sample_rate=16000):
        """Apply eval-only audio perturbations for the audio shift test."""
        mode = self.audio_shift_mode
        if mode in (None, "aligned"):
            return audio

        if mode.startswith("shift_"):
            shift_spec = mode[len("shift_"):]
            shift_seconds = (
                float(shift_spec[:-1]) if shift_spec.endswith("s") else float(shift_spec)
            )
            shift_samples = int(round(shift_seconds * sample_rate))
            return torch.roll(audio, shifts=shift_samples, dims=0)

        if mode == "shuffle":
            chunk_samples = max(1, int(round(self.audio_shuffle_chunk_seconds * sample_rate)))
            if audio.shape[0] <= chunk_samples:
                return audio
            chunks = [
                audio[start : min(start + chunk_samples, audio.shape[0])]
                for start in range(0, audio.shape[0], chunk_samples)
            ]
            seed = int(hashlib.md5(f"{qid}_{mode}".encode("utf-8")).hexdigest()[:8], 16)
            generator = torch.Generator()
            generator.manual_seed(seed)
            order = torch.randperm(len(chunks), generator=generator).tolist()
            shuffled_audio = torch.cat([chunks[idx] for idx in order], dim=0)
            return shuffled_audio[: audio.shape[0]]

        raise ValueError(f"Unsupported audio_shift_mode: {mode}")

    def apply_scene_boundary_eval_perturbation(
        self, scene_start_times, scene_start_frames, qid, total_frames, fps
    ):
        """Apply eval-only shot-boundary perturbations for robustness analysis."""
        mode = self.scene_boundary_mode
        if mode in (None, "default"):
            return scene_start_times, scene_start_frames

        if total_frames <= 0 or fps <= 0:
            return scene_start_times, scene_start_frames

        if mode == "pseudo_shots":
            step = max(1, int(round(self.pseudo_shot_seconds * fps)))
            starts = list(range(0, total_frames, step))
            times = [start / fps for start in starts]
            return times, starts

        if not scene_start_frames:
            return scene_start_times, scene_start_frames

        seed = int(hashlib.md5(f"{qid}_{mode}".encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator()
        generator.manual_seed(seed)

        starts = list(scene_start_frames)
        if starts[0] != 0:
            starts = [0] + starts

        if mode == "perturb_5":
            if len(starts) <= 1:
                return scene_start_times, starts
            perturbed = [starts[0]]
            max_frame = max(0, total_frames - 1)
            for frame in starts[1:]:
                offset = torch.randint(
                    -self.scene_boundary_perturb_frames,
                    self.scene_boundary_perturb_frames + 1,
                    (1,),
                    generator=generator,
                ).item()
                perturbed.append(min(max(frame + offset, 0), max_frame))
            perturbed = sorted(set(perturbed))
            if perturbed[0] != 0:
                perturbed = [0] + perturbed
            times = [frame / fps for frame in perturbed]
            return times, perturbed

        if mode == "corrupt_20":
            if len(starts) <= 2:
                return scene_start_times, starts
            interior = starts[1:]
            keep_mask = torch.ones(len(interior), dtype=torch.bool)
            drop_count = int(round(len(interior) * self.scene_boundary_corruption_ratio))
            if drop_count > 0:
                drop_indices = torch.randperm(len(interior), generator=generator)[:drop_count]
                keep_mask[drop_indices] = False
            corrupted = [0] + [frame for keep, frame in zip(keep_mask.tolist(), interior) if keep]
            times = [frame / fps for frame in corrupted]
            return times, corrupted

        raise ValueError(f"Unsupported scene_boundary_mode: {mode}")

    def process_video(self, video_path, save_dir, n_frms):
        os.makedirs(save_dir, exist_ok=True)

        vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
        vlen = len(vr) 
        fps = vr.get_avg_fps()  
        if not fps or fps <= 0:
            raise ValueError("Could not determine a valid video frame rate")

        video_name = os.path.basename(video_path)

        if video_name in self.scene_data:
            scene_start_times = self.scene_data[video_name]["scene_start_times"]
            scene_start_frames = self.scene_data[video_name]["scene_start_frames"]
            scene_list = [
                (FrameTimecode(timecode=scene_start_times[i], fps=fps), 
                FrameTimecode(timecode=scene_start_times[i + 1], fps=fps))
                for i in range(len(scene_start_times) - 1)
            ]
            if scene_start_times:
                scene_list.append((
                    FrameTimecode(timecode=scene_start_times[-1], fps=fps), 
                    FrameTimecode(timecode=vlen / fps, fps=fps)
                ))
        else:
            scene_list = []
        if not scene_list:
            scene_list = []
            total_duration = vlen / fps
            start_time = 0.0
            while start_time < total_duration:
                end_time = min(start_time + 3.0, total_duration)
                scene_list.append((FrameTimecode(timecode=start_time, fps=fps), 
                                FrameTimecode(timecode=end_time, fps=fps)))
                start_time = end_time

        all_frms, all_frame_indices, all_scene_labels, all_scene_indices = [], [], [], []
        scene_start_times, scene_start_frames = [], [] 

        for scene_id, (start_time, end_time) in enumerate(scene_list, start=0):
            start_frame, end_frame = start_time.get_frames(), end_time.get_frames()
            duration = end_time.get_seconds() - start_time.get_seconds()
            scene_start_times.append(start_time.get_seconds())
            scene_start_frames.append(start_frame)
            selected_frames = []
            if 1 <= duration < 2:
                selected_frames.append((start_frame + end_frame) // 2)
                all_scene_indices.append(start_frame)
            else:
                seg_start = start_frame
                while seg_start + int(fps * 2) <= end_frame:
                    seg_end = seg_start + int(fps * 2)
                    mid_frame = (seg_start + seg_end) // 2
                    selected_frames.append(mid_frame)
                    all_scene_indices.append(mid_frame)
                    seg_start = seg_end
            if not selected_frames:
                continue 
            selected_frames = [min(max(0, f), vlen - 1) for f in selected_frames]  
            all_frame_indices.extend(selected_frames)
            all_scene_labels.extend([scene_id] * len(selected_frames))

        return scene_start_times, scene_start_frames
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])

            clip = [start, end]
        else:
            clip = None

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")
        # Preserve the historical first sampling pass and its training RNG consumption.
        n_frms = self.vis_processor(video_path, clip_proposal=clip)
        frms, indices, fps, n_frms = self.vis_processor(video_path, clip_proposal=clip)

        scene_start_times, scene_start_frames = self.process_video(video_path, 'video_test', n_frms)
        scene_start_times, scene_start_frames = self.apply_scene_boundary_eval_perturbation(
            scene_start_times,
            scene_start_frames,
            ann["qid"],
            len(VideoReader(uri=video_path)),
            fps,
        )

        scene_labels = []
        for frame in indices:
            scene_index = bisect.bisect_right(scene_start_frames, frame) - 1
            scene_labels.append(scene_index)
        time_indices = [round(index / fps, 1) for index in indices]
        time_indices = torch.tensor(time_indices)
        if time_indices.size(0) < n_frms:
            time_indices = torch.cat([time_indices, time_indices[-1].repeat(n_frms - time_indices.size(0))])

        audio = self.extract_audio(video_path)
        audio = self.apply_audio_eval_perturbation(audio, ann["qid"])

        query = ann["query"]

        relevant_windows = str(ann["relevant_windows"])
        query_prompt = f"Query: {query}. Use both visual and auditory information to find the moment.\n"
        task_prompt = (
            "Given the video and the query, use both visual and auditory information to identify "
            "the index range corresponding to the moment described.\nIndex range: "
        )

        frms = frms.permute(1, 0, 2, 3)
        time_stamps = [float(idx / fps) for idx in indices]
        duration = ann["duration"]
        timestamps = [round(t, 2) for t in time_stamps]
        timestamps = torch.tensor(timestamps)
        duration = torch.tensor(duration)

        video_prompt_end = "<extra_id_0>"
        time_begin_token = "<extra_id_1>"
        time_end_token = "<extra_id_2>" 
        frame_begin_token = "<extra_id_3>"
        frame_end_token = "<extra_id_4>"        
        audio_prompt_end = "<extra_id_5>"

        return {
            "video": frms,
            "scene_labels": torch.tensor(scene_labels),
            "duration": duration,
            "query_id": ann["qid"],
            "video_name": ann["video"],
            "query_text": query,
            "frame_indices": torch.tensor(indices, dtype=torch.long),
            "n_frames": torch.tensor(n_frms, dtype=torch.long),

            "scene_start_frames_debug": json.dumps(scene_start_frames),
            "scene_start_times_debug": json.dumps(scene_start_times),
            "timestamps": timestamps,
            "video_prompt_end": video_prompt_end,
            "time_begin_token": time_begin_token,
            "time_end_token": time_end_token,
            "frame_begin_token": frame_begin_token,
            "frame_end_token": frame_end_token,
            "query_prompt": query_prompt,
            "task_prompt": task_prompt,
            "relevant_windows": relevant_windows,
            "audio":audio,
            "audio_prompt_end": audio_prompt_end
        }
