import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from components import models, datasets, methods
from utils.logger import ExpLogger, TimeCheck
from utils.metric import SummationMeter, Metric


class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        # 仅 master 进程创建 logger
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None
        if self.cfg is not None:
            self._init_from_cfg(cfg)
        self.current_epoch = 0

    # --------------------------------------------------------------------- #
    # 初始化
    # --------------------------------------------------------------------- #
    def _init_from_cfg(self, cfg):
        assert cfg is not None
        self.cfg = cfg

        self.model = _prepare_model(
            self.cfg.MODEL,
            is_distributed=self.args.is_distributed,
            local_rank=self.args.local_rank if self.args.is_distributed else None
        )
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)

        self.get_train_loader = getattr(datasets, self.cfg.DATASET.TRAIN.NAME).get_dataloader
        self.get_test_loader  = getattr(datasets, self.cfg.DATASET.TEST .NAME).get_dataloader
        self.method = getattr(methods, self.cfg.METHOD)

    # --------------------------------------------------------------------- #
    # 训练
    # --------------------------------------------------------------------- #
    def train(self):
        if self.args.is_master:
            self._log_before_train()

        train_loader = self.get_train_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.TRAIN,
            dataloader_cfg=self.cfg.DATALOADER.TRAIN,
            is_distributed=self.args.is_distributed
        )

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()

        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)

            train_log_dict = self.method.train(
                model=self.model,
                data_loader=train_loader,
                optimizer=self.optimizer,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size
            )

            self.scheduler.step()
            self.current_epoch += 1

            if self.args.is_distributed:
                train_log_dict = self._gather_log(train_log_dict)

            if self.args.is_master:
                self._log_after_epoch(epoch + 1, time_checker, train_log_dict, 'train')

    # --------------------------------------------------------------------- #
    # 推理 / 测试
    # --------------------------------------------------------------------- #
    def test(self):
        test_loader = self.get_test_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.TEST,
            dataloader_cfg=self.cfg.DATALOADER.TEST
        )

        if self.args.is_master and self.logger:
            self.logger.test()

        for sequence_dataloader in test_loader:
            sequence_name = sequence_dataloader.dataset.sequence_name
            sequence_pred_list = self.method.test(
                model=self.model,
                data_loader=sequence_dataloader
            )

            # 仅 master 保存可视化
            if self.args.is_master and self.logger:
                for cur_pred_dict in sequence_pred_list:
                    file_name = cur_pred_dict.pop('file_name')
                    for key in cur_pred_dict:
                        self.logger.save_visualize(
                            image=cur_pred_dict[key],
                            visual_type=key,
                            sequence_name=os.path.join('test', sequence_name),
                            image_name=file_name
                        )

    # --------------------------------------------------------------------- #
    # Checkpoint 相关
    # --------------------------------------------------------------------- #
    def save(self, name):
        if self.logger:
            self.logger.save_checkpoint(self._make_checkpoint(), name)

    def load(self, name):
        """
        仅使用 torch.load 加载 checkpoint，避免非 master 进程 logger 为 None 报错
        """
        assert os.path.isfile(name), f"Checkpoint file not found: {name}"
        checkpoint = torch.load(name, map_location=f'cuda:{self.args.local_rank}')
        self._init_from_cfg(checkpoint['cfg'])        # 先恢复网络结构
        self.model.module.load_state_dict(checkpoint['model'], strict=False)

        if self.args.is_master:
            print(f"Checkpoint is Loaded from {name}")

    # --------------------------------------------------------------------- #
    # 内部工具
    # --------------------------------------------------------------------- #
    def _make_checkpoint(self):
        return {
            'epoch':     self.current_epoch,
            'args':      self.args,
            'cfg':       self.cfg,
            'model':     self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def _gather_log(self, log_dict):
        if log_dict is None:
            return None
        for key in log_dict.keys():
            if isinstance(log_dict[key], (SummationMeter, Metric)):
                log_dict[key].all_gather(self.args.world_size)
        return log_dict

    def _log_before_train(self):
        if not self.logger:
            return
        self.logger.train()
        self.logger.save_args(self.args)
        self.logger.save_cfg(self.cfg)
        self.logger.log_model(self.model)
        self.logger.log_optimizer(self.optimizer)
        self.logger.save_src(os.path.dirname(os.path.abspath(__file__)))

    def _log_after_epoch(self, epoch, time_checker, log_dict, part):
        if not self.logger:
            return
        # 时间
        time_checker.update(epoch)
        self.logger.write(f'Epoch: {epoch} | time/epoch: {time_checker.time_per_epoch} | eta: {time_checker.eta}')
        # 日志
        log = f'{part:5s}'
        for key in log_dict.keys():
            log += f' | {key}: {log_dict[key]}'
            if isinstance(log_dict[key], (SummationMeter, Metric)):
                self.logger.add_scalar(f'{part}/{key}', log_dict[key].value, epoch)
            else:
                self.logger.add_scalar(f'{part}/{key}', log_dict[key], epoch)
        self.logger.write(log)
        # 保存 ckpt
        ckpt = self._make_checkpoint()
        self.logger.save_checkpoint(ckpt, 'final.pth')
        if epoch % self.args.save_term == 0:
            self.logger.save_checkpoint(ckpt, f'{epoch}.pth')


# ------------------------------------------------------------------------- #
# 构建网络 / 优化器 / 调度器
# ------------------------------------------------------------------------- #
def _prepare_model(model_cfg, is_distributed=False, local_rank=None):
    model = getattr(models, model_cfg.NAME)(**model_cfg.PARAMS)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = nn.DataParallel(model).cuda()
    return model


def _prepare_optimizer(optimizer_cfg, model):
    params_group = model.module.get_params_group(optimizer_cfg.PARAMS.lr)
    optimizer = getattr(optim, optimizer_cfg.NAME)(params_group, **optimizer_cfg.PARAMS)
    return optimizer


def _prepare_scheduler(scheduler_cfg, optimizer):
    if scheduler_cfg.NAME == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_cfg.PARAMS)
    else:
        scheduler = getattr(optim.lr_scheduler, scheduler_cfg.NAME)(optimizer, **scheduler_cfg.PARAMS)
    return scheduler
