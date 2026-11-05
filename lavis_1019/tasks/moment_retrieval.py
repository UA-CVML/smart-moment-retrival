import logging
import json
import os
import numpy as np
import ast
import re
import time
from collections import OrderedDict, defaultdict
import multiprocessing as mp

import torch
import torch.distributed as dist
import wandb

from lavis_1019.common.registry import registry
from lavis_1019.common.logger import MetricLogger, SmoothedValue
from lavis_1019.tasks.base_task import BaseTask
from lavis_1019.common.dist_utils import (
    main_process,
    is_dist_avail_and_initialized,
    is_main_process,
)
from lavis_1019.datasets.data_utils import prepare_sample

from lavis_1019.tasks.mr_eval import eval_submission

@registry.register_task("moment_retrieval")
class MomentRetrievalTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.report_metric = True

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        task = cls()
        task.report_metric = cfg.run_cfg.get("report_metric", True)
        return task

    def _sync_cuda(self, cuda_enabled):
        if cuda_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_batch_size(self, samples):
        if "query_id" in samples:
            query_id = samples["query_id"]
            if isinstance(query_id, (list, tuple)):
                return len(query_id)
            return 1

        return 1

    def valid_step(self, model, samples):
        results = []
        batch_size = self._get_batch_size(samples)

        self._sync_cuda(cuda_enabled=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model_start_time = time.perf_counter()
        outputs = model.generate(samples)
        self._sync_cuda(cuda_enabled=True)
        model_latency_ms = (time.perf_counter() - model_start_time) * 1000.0
        peak_gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        answer = outputs["answer"]
        qid = outputs["qid"]
        pred = outputs["prediction"]
        raw_pred = outputs["raw_prediction"]
        duration = outputs["duration"]
        token_stats = outputs.get("token_stats", {})
        tokens_before = token_stats.get("visual_tokens_before", [])
        tokens_after = token_stats.get("visual_tokens_after", [])
        tokens_kept_pct = token_stats.get("visual_tokens_kept_pct", [])
        assert len(qid) == len(answer)
        assert len(qid) == len(pred)

        i = 0
        for a, q, p, rp, d, tb, ta, tk in zip(
            answer, qid, pred, raw_pred, duration, tokens_before, tokens_after, tokens_kept_pct
        ):
            results.append(
                {
                    "qid": q + "_" + str(i),
                    "raw_prediction": rp,
                    "prediction": p,
                    "target": a,
                    "duration": d,
                    "visual_tokens_before": tb,
                    "visual_tokens_after": ta,
                    "visual_tokens_kept_pct": tk,
                    "model_latency_ms": model_latency_ms / batch_size,
                    "peak_gpu_mem_gb": peak_gpu_mem_gb,
                }
            )
            i += 1

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )

        if not self.report_metric:
            logging.info("Predictions saved to %s; metric computation disabled.", eval_result_file)
            return {}

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        if is_main_process() and wandb.run is not None:

            wandb.log(
                {
                    "eval/agg_metrics": metrics["agg_metrics"],

                    "eval/R1@0.5": metrics["r1"]["0.5"],
                    "eval/R1@0.7": metrics["r1"]["0.7"],

                    "eval/mAP@0.5": metrics["mAP"]["0.5"],
                    "eval/mAP@0.75": metrics["mAP"]["0.75"],
                    "eval/mIoU": metrics["mIoU"],
                    "eval/visual_tokens_before_mean": metrics["visual_tokens_before_mean"],
                    "eval/visual_tokens_after_mean": metrics["visual_tokens_after_mean"],
                    "eval/visual_tokens_kept_pct_mean": metrics["visual_tokens_kept_pct_mean"],
                    "eval/model_latency_ms_mean": metrics["model_latency_ms_mean"],
                    "eval/end_to_end_latency_ms_mean": metrics["end_to_end_latency_ms_mean"],
                    "eval/throughput_samples_per_s_mean": metrics["throughput_samples_per_s_mean"],
                    "eval/peak_gpu_mem_gb_mean": metrics["peak_gpu_mem_gb_mean"],
                    "eval/invalid_predictions": metrics["invalid_predictions"],
                }
            )

        return metrics

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"

        print_freq = 10

        results = []
        i = 0
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            end_to_end_start_time = time.perf_counter()
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            samples.update({"iters": i})

            eval_output = self.valid_step(model=model, samples=samples)
            self._sync_cuda(cuda_enabled=cuda_enabled)
            end_to_end_latency_ms = (time.perf_counter() - end_to_end_start_time) * 1000.0
            batch_size = max(len(eval_output), 1)
            end_to_end_latency_per_sample_ms = end_to_end_latency_ms / batch_size
            throughput_samples_per_s = batch_size / (end_to_end_latency_ms / 1000.0)
            for result in eval_output:
                result["end_to_end_latency_ms"] = end_to_end_latency_per_sample_ms
                result["throughput_samples_per_s"] = throughput_samples_per_s
            results.extend(eval_output)
            i += 1

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        """Report metrics to the logger and save to the output directory.
        Compute r1 for the moment retrieval task.
        Args:
            eval_result_file (str): path to the evaluation result file.
            split_name (str): split name, e.g. val, test
        """
        results = json.load(open(eval_result_file))
        total_num = len(results)
        visual_tokens_before_mean = sum(
            r.get("visual_tokens_before", 0) for r in results
        ) / total_num
        visual_tokens_after_mean = sum(
            r.get("visual_tokens_after", 0) for r in results
        ) / total_num
        visual_tokens_kept_pct_mean = sum(
            r.get("visual_tokens_kept_pct", 0.0) for r in results
        ) / total_num
        model_latency_ms_mean = sum(
            r.get("model_latency_ms", 0.0) for r in results
        ) / total_num
        peak_gpu_mem_gb_mean = sum(
            r.get("peak_gpu_mem_gb", 0.0) for r in results
        ) / total_num
        end_to_end_latency_ms_mean = sum(
            r.get("end_to_end_latency_ms", 0.0) for r in results
        ) / total_num
        throughput_samples_per_s_mean = sum(
            r.get("throughput_samples_per_s", 0.0) for r in results
        ) / total_num

        results_interpreted = [
            {
                "qid": r["qid"],
                "pred_relevant_windows": self.moment_str_to_list(r["prediction"]),
                "relevant_windows": self.moment_str_to_list(r["target"]),
            }
            for r in results
        ]

        all_metrics = eval_submission(results_interpreted, results_interpreted)

        r1_avg = all_metrics["brief"]["MR-full-R1-avg"]
        mIoU = all_metrics["brief"]["MR-full-mIoU"]
        invalid_pred_num = all_metrics["brief"]["MR-full-invalid_pred_num"]

        metrics = {
            "agg_metrics": r1_avg,
            "r1": all_metrics["full"]["MR-R1"],
            "mAP": all_metrics["full"]["MR-mAP"],
            "mIoU": mIoU,
            "visual_tokens_before_mean": visual_tokens_before_mean,
            "visual_tokens_after_mean": visual_tokens_after_mean,
            "visual_tokens_kept_pct_mean": visual_tokens_kept_pct_mean,
            "model_latency_ms_mean": model_latency_ms_mean,
            "peak_gpu_mem_gb_mean": peak_gpu_mem_gb_mean,
            "end_to_end_latency_ms_mean": end_to_end_latency_ms_mean,
            "throughput_samples_per_s_mean": throughput_samples_per_s_mean,
            "invalid_predictions": invalid_pred_num / total_num,
            "total": total_num,
        }

        logging.info(metrics)
        return metrics

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):

            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:

            inner_epoch = epoch
        else:

            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):

            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if is_main_process() and wandb.run is not None:
                wandb.log(
                    {
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def moment_str_to_list(self, m):
        """Convert a string of moments to a list of moments.
        If predicted string is not a list, it means that the model has not yet learned to predict the right format.
        In that case, we return [[-1, -1]] to represent an error.
        This will then lead to an IoU of 0.
        Args:
            m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
        Returns:
            list: a list of moments, e.g. [[0, 1], [4, 7]]
        """
        if m == "[[-1, -1]]":
            return [[-1, -1]]

        if not re.match(r"\[\[.*\]\]", m):
            return [[-1, -1]]

        try:
            _m = ast.literal_eval(m)
        except:
            return [[-1, -1]]

        if not isinstance(_m, list):

            return [[-1, -1]]

        for i in range(len(_m)):

            if len(_m[i]) != 2:

                _m[i] = [-1, -1]

        return _m
