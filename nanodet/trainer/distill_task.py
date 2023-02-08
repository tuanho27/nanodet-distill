import os
import numpy as np
from tqdm import tqdm
import copy
import json
import warnings
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

import torch.distributed as dist

from nanodet.data.batch_process import stack_batch_img
from nanodet.util import convert_avg_params, gather_results, mkdir

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
from .rkd_distiller import RKDDistill


class TrainingDistillTask(pl.LightningModule):
    """
    Distill the knowledge from a teacher model to a student model.
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingDistillTask, self).__init__()
        self.cfg = cfg
        self.temperature = self.cfg.distill.temperature
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)       
        self.teacher_model = build_model(cfg.teacher_model) # Set teacher models
        self.softmax = torch.nn.Softmax(1)
        self.conv1x1 = torch.nn.Conv2d(cfg.teacher_model.arch.fpn.in_channels[-1], cfg.model.arch.fpn.in_channels[-1], 1)
        self.conv1x12 = torch.nn.Conv2d(cfg.teacher_model.arch.fpn.in_channels[-2], cfg.model.arch.fpn.in_channels[-2], 1)
        self.triplet = F.cosine_embedding_loss
        self.lambda_student = 0.7
        self.T_student = 2.0
        self.rkd = RKDDistill()
        # self.kd_fun = torch.nn.KLDivLoss(size_average=False).to(self.device)
    
    def kd_triplet_loss(self, out_s, out_t, target):
        lambda_ = self.lambda_student
        T = self.T_student
        # Standard Learning Loss ( Classification Loss)
        # loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        batch_size = len(target)
        # target = torch.cat(target)
        s_max = F.log_softmax(out_s / T, dim=1).to(self.device)
        t_max = F.softmax(out_t / T, dim=1).to(self.device)
        # pred_s = out_s.data.max(1, keepdim=True)[1]
        # pred_t = out_t.data.max(1, keepdim=True)[1]
        y = torch.ones(len(torch.cat(target))).to(self.device)
        # loss = self.triplet(out_s, out_t, y)
        # loss_kd = self.kd_fun(s_max, t_max) / batch_size
        loss = F.cosine_embedding_loss(out_s, out_t, y)
        loss_kd = F.kl_div(self.softmax(out_t), self.softmax(out_t), reduction='batchmean') * (self.T_student**2) + 0.1
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        return loss

    def weighted_kl_div(self, ps, qt):
        eps = 1e-10
        ps = ps + eps
        qt = qt + eps
        log_p = qt * torch.log(ps)
        log_p[:, 0] *= self.neg_w
        log_p[:, 1:] *= self.pos_w
        return -torch.sum(log_p)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, batch):
        """
        One forward pass of knowledge distillation training

        """

        self.teacher_model.eval()
        with torch.no_grad():
            tc_preds, tc_feat, tc_loss, tc_loss_states, _ = self.teacher_model.forward_train(batch)
            teacher_prob, tc_bbox_preds = tc_preds.split(
                [self.cfg.teacher_model.arch.head.num_classes, 4 * (self.cfg.teacher_model.arch.head.reg_max + 1)], dim=-1
                )
            teacher_prob = teacher_prob.reshape(-1, self.cfg.model.arch.head.num_classes)
            teacher_logits = tc_feat[-1]

        self.model.train()
        preds, feat, loss, loss_states, prior_assigns = self.model.forward_train(batch)
        student_prob, st_bbox_preds = preds.split(
                [self.cfg.model.arch.head.num_classes, 4 * (self.cfg.model.arch.head.reg_max + 1)], dim=-1
            )
        student_prob = student_prob.reshape(-1, self.cfg.model.arch.head.num_classes)
        student_logits = feat[-1]
        bz = student_logits.shape[0]
        final_state_loss = loss_states
        # print(f"TC prob: {teacher_prob.shape} ST prob: {student_prob.shape}")
        # print(f"TC logit: {teacher_logits.shape} ST logit: {student_logits.shape}")

        kl_loss  = F.kl_div(self.softmax(student_prob), self.softmax(teacher_prob), reduction='batchmean') * (self.temperature**2) + 0.1
        embed_loss = 1 - torch.mean(F.cosine_similarity(student_logits, self.conv1x1(teacher_logits), dim=-1)) #this loss will be small if two tensor are similar
        # kd_triplet_loss = self.kd_triplet_loss(student_prob, teacher_prob, prior_assigns[0])
        rkd_loss = self.rkd.calculate_loss([self.conv1x1(tc_feat[-1]), self.conv1x12(tc_feat[-2])],
                                            F.adaptive_avg_pool2d(teacher_logits, 1).view(bz,-1), 
                                            [feat[-1], feat[-2]], 
                                            F.adaptive_avg_pool2d(student_logits, 1).view(bz,-1))

        final_state_loss["kd_loss"] = kl_loss
        final_state_loss["cos_embed_loss"] = embed_loss
        # final_state_loss['kd_triplet_loss'] = self.kd_triplet_loss(student_prob, teacher_prob, prior_assigns[0])
        final_state_loss["rkd_loss"] = rkd_loss

        # final_loss = loss + kd_triplet_loss #+ kl_loss + embed_loss
        final_loss = loss + kl_loss + embed_loss + rkd_loss

        logit_diff = F.mse_loss(student_logits, self.conv1x1(teacher_logits))
        prob_diff = F.l1_loss(self.softmax(student_prob), self.softmax(teacher_prob))

        return final_loss, final_state_loss

    def forward_test(self, x):
        x, _ = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        preds, _ = self.forward_test(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)

        loss, loss_states = self(batch)

        # log train losses
        if self.global_step % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )

            self.logger.info(log_msg)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
        self.lr_scheduler.step()

   
    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        if self.weight_averager is not None:
            preds, feat, loss, loss_states, _ = self.avg_model.forward_train(batch)
        else:
            preds, feat, loss, loss_states, _ = self.model.forward_train(batch)

        if batch_idx % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        return dets


    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))
            
    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop("name")
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        # lr_scheduler = {'scheduler': self.lr_scheduler,
        #                 'interval': 'epoch',
        #                 'frequency': 1}
        # return [optimizer], [lr_scheduler]

        return optimizer

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                warmup_lr = (
                    self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
                )
            elif self.cfg.schedule.warmup.name == "linear":
                k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (
                    1 - self.cfg.schedule.warmup.ratio
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * k
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)
        
    # ------------Hooks-----------------
    def on_train_start(self):
        if self.current_epoch > 0:
            self.lr_scheduler.last_epoch = self.current_epoch - 1

    def on_pretrain_routine_end(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_epoch_start(self):
        self.model.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self):
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")
