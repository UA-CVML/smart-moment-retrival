"""CPU checks for release paths and inference without public test labels."""
import json
import importlib
import os
import sys
import tempfile
import unittest
import torch
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from lavis_1019.common.registry import registry
from lavis_1019.runners.runner_base import RunnerBase
from lavis_1019.datasets.datasets.moment_retrieval_dataset import MomentRetrievalDataset
from lavis_1019.tasks.moment_retrieval import MomentRetrievalTask
from lavis_1019.common.config import Config
from lavis_1019.models.blip2_mr_models.blip2_mr import BLIP2_MR


class ReleaseTests(unittest.TestCase):
    def test_public_registry_contains_only_smart_qvhighlights(self):
        self.assertEqual(registry.list_models(), ['blip2_mr'])
        self.assertEqual(registry.list_datasets(), ['qvh'])
        self.assertEqual(registry.list_tasks(), ['moment_retrieval'])
        self.assertEqual(list(BLIP2_MR.PRETRAINED_MODEL_CONFIG_DICT), ['pretrain_flant5xl'])

    def test_all_public_configs_select_full_smart(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ['train', 'eval', 'test']:
            with self.subTest(mode=mode):
                cfg = Config(SimpleNamespace(
                    cfg_path=str(root / 'configs/qvhighlights' / (mode + '.yaml')),
                    options=[],
                ))
                self.assertEqual(cfg.model_cfg.arch, 'blip2_mr')
                self.assertEqual(cfg.model_cfg.stc_ablation_mode, 'full_stc')
                self.assertTrue(cfg.model_cfg.use_audio)
                self.assertTrue(cfg.model_cfg.interleave_data)
                self.assertEqual(cfg.model_cfg.task, 'qformer_freeze_lora')
                self.assertEqual(cfg.model_cfg.keyframe_topk, 32)
                self.assertEqual(cfg.datasets_cfg.qvh.vis_processor.eval.n_frms, 80)
                self.assertEqual(cfg.run_cfg.task, 'moment_retrieval')

    def test_entrypoints_use_repo_relative_paths_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                for module_name, mode in [('train', 'train'), ('evaluate', 'eval')]:
                    with self.subTest(entrypoint=module_name), ExitStack() as stack:
                        entrypoint = importlib.import_module(module_name)
                        os.chdir(directory)
                        stack.enter_context(patch.object(sys, 'argv', [
                            module_name + '.py', '--cfg-path',
                            'configs/qvhighlights/' + mode + '.yaml',
                        ]))
                        for name in ['init_distributed_mode', 'setup_seeds', 'setup_logger']:
                            stack.enter_context(patch.object(entrypoint, name))
                        stack.enter_context(patch.object(Config, 'pretty_print'))
                        task = MagicMock()
                        setup_task = stack.enter_context(patch.object(
                            entrypoint.tasks, 'setup_task', return_value=task,
                        ))
                        runner = MagicMock()
                        if mode == 'train':
                            stack.enter_context(patch.object(
                                entrypoint, 'get_runner_class', return_value=runner,
                            ))
                        else:
                            stack.enter_context(patch.object(entrypoint, 'RunnerBase', runner))
                        entrypoint.main()
                        self.assertEqual(Path.cwd(), root)
                        cfg = setup_task.call_args.args[0]
                        annotation = cfg.datasets_cfg.qvh.build_info.annotations.val.storage
                        self.assertFalse(Path(annotation).is_absolute())
                        self.assertTrue(Path(annotation).is_file())
                        self.assertEqual(
                            Path(cfg.model_cfg.beats_checkpoint).resolve(),
                            root / 'checkpoints/BEATs_iter3.pt',
                        )
                        task.build_model.assert_called_once()
                        runner.assert_called_once()
        finally:
            os.chdir(original_cwd)

    def test_output_directory_is_relative_to_repo_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'moved-checkout'
            with patch.dict(registry.mapping['paths'], {
                'repo_root': str(root), 'library_root': str(root / 'lavis_1019'),
            }, clear=True):
                cfg = SimpleNamespace(run_cfg=SimpleNamespace(output_dir='output/qvhighlights/train'))
                runner = RunnerBase(cfg=cfg, task=None, model=None, datasets={}, job_id='test')
                self.assertEqual(runner.output_dir, root / 'output/qvhighlights/train')
                self.assertTrue((runner.output_dir / 'result').is_dir())

    def test_dataset_reads_configured_shot_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / 'annotations.json'
            annotations.write_text('[]')
            scenes = {'clip.mp4': {'scene_start_times': [0.0, 2.5],
                                   'scene_start_frames': [0, 75]}}
            scene_file = root / 'scenes.json'
            scene_file.write_text(json.dumps(scenes))
            dataset = MomentRetrievalDataset(
                vis_processor=None, text_processor=None, vis_root=directory,
                ann_paths=[str(annotations)], scene_data_file=str(scene_file),
            )
            self.assertEqual(dataset.scene_data, scenes)

    def test_full_stc_keeps_keyframes_and_filters_other_frames(self):
        model = BLIP2_MR.__new__(BLIP2_MR)
        torch.nn.Module.__init__(model)
        model.stc_ablation_mode = 'full_stc'
        frame = torch.arange(64, dtype=torch.float32).reshape(32, 2)
        positions = list(range(0, 32, 2))
        shot = {'count': 2, 'qformer_output': positions}
        self.assertTrue(torch.equal(model.apply_stc_ablation_to_frame(frame, 1, shot, 2), frame))
        self.assertTrue(torch.equal(model.apply_stc_ablation_to_frame(frame, 0, shot, 0), frame))
        filtered = model.apply_stc_ablation_to_frame(frame, 0, shot, 2)
        self.assertTrue(torch.equal(filtered, frame[positions]))
        singleton = model.apply_stc_ablation_to_frame(frame, 0, {'count': 1}, 2)
        self.assertEqual(tuple(singleton.shape), (24, 2))

    def test_audio_pooling_keeps_150_temporal_tokens(self):
        model = BLIP2_MR.__new__(BLIP2_MR)
        torch.nn.Module.__init__(model)
        features = torch.arange(600, dtype=torch.float32).reshape(1, 300, 2)
        pooled = model.compress_audio_features(features)
        self.assertEqual(tuple(pooled.shape), (1, 150, 2))
        self.assertTrue(torch.equal(pooled, features.reshape(1, 150, 2, 2).mean(dim=2)))

    def test_test_inference_saves_predictions_without_scoring_dummy_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            registry.register_path('result_dir', directory)
            task = MomentRetrievalTask.setup_task(
                cfg=SimpleNamespace(run_cfg={'report_metric': False})
            )
            rows = [{'qid': 'QVHighlight_1_0', 'prediction': '[[1, 3]]'}]
            metrics = task.after_evaluation(rows, 'test', 'best')
            self.assertEqual(metrics, {})
            self.assertEqual(json.loads((Path(directory) / 'test_epochbest.json').read_text()), rows)


if __name__ == '__main__':
    unittest.main()
