import os
import argparse
import time
import torch
import torch.distributed as dist
import torchprofile

from manager import DLManager
from utils.config import get_cfg


def get_nested(cfg_obj, keys, default=None):
    cur = cfg_obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return cur if cur is not None else default


def measure_inference_speed(model, device, dummy_input, warmup=10, test_runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(left_event=dummy_input, right_event=dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(test_runs):
            _ = model(left_event=dummy_input, right_event=dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / test_runs
    fps = 1.0 / avg_time if avg_time > 0 else float('inf')
    mem_used = torch.cuda.max_memory_allocated() / 1024**2
    return avg_time, fps, mem_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/root/code/configs/config.yaml')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    args.is_distributed = True
    args.is_master = args.local_rank == 0
    args.device = f'cuda:{args.local_rank}'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()

    assert os.path.isfile(args.config_path)
    assert os.path.isdir(args.data_root)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    cfg = get_cfg(args.config_path)
    exp_manager = DLManager(args, cfg)
    exp_manager.load(args.checkpoint_path)

    if args.is_master:
        crop_h = get_nested(cfg, ['DATASET', 'TEST', 'PARAMS', 'crop_height'], 256)
        crop_w = get_nested(cfg, ['DATASET', 'TEST', 'PARAMS', 'crop_width'], 256)
        stack_size = get_nested(cfg, ['DATASET', 'TEST', 'PARAMS', 'event_cfg', 'PARAMS', 'stack_size'], 5)

        # 新增：构造符合模型要求的 6D 输入 (B, C, H, W, T, S)
        dummy_input = torch.randn(1, 1, crop_h, crop_w, stack_size, 1).to(args.device)

        model = exp_manager.model.module
        model.eval()

        # FLOPs
        try:
            flops = torchprofile.profile_macs(model, (dummy_input, dummy_input)) * 2
            print(f"[MASTER] FLOPs: {flops / 1e9:.2f} GFLOPs")
        except Exception as e:
            print("[MASTER] Failed to measure FLOPs:", e)

        # 参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MASTER] Params: {total_params / 1e6:.2f} M")

        # 运行时间 / FPS / 显存
        avg_t, fps, mem = measure_inference_speed(model, torch.device(args.device), dummy_input)
        print(f"[MASTER] Avg time per frame: {avg_t:.4f} s | FPS: {fps:.2f} | Max Mem: {mem:.2f} MB")

    exp_manager.test()


if __name__ == '__main__':
    main()
