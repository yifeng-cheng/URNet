import os
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from manager import DLManager

# Optional: FLOPs/params via thop
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[INFO] 'thop' not found – FLOPs/Params computation will be skipped. Install via `pip install thop` if needed.")


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def compute_flops_params(model, input_shape, device):
    """Return FLOPs (MACs) and parameter count for a single forward pass.

    Args:
        model: torch.nn.Module – already moved to the correct device.
        input_shape: list/tuple – (C, H, W).
        device: torch.device or int.
    Returns:
        (flops, params) – formatted strings, or (None, None) if THOP unavailable.
    """
    if not HAS_THOP:
        return None, None

    dummy = torch.randn(1, *input_shape, device=device)
    model.eval()

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)

    flops, params = map(lambda x: clever_format([x], "%.3f")[0], (flops, params))
    return flops, params


def measure_runtime_memory(func):
    """Decorator to measure runtime (ms) and peak GPU memory (MB) of a function."""

    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            runtime_ms = start_event.elapsed_time(end_event)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
            return result, runtime_ms, peak_mem
        else:
            t0 = time.time()
            result = func(*args, **kwargs)
            runtime_ms = (time.time() - t0) * 1000
            peak_mem = None
            return result, runtime_ms, peak_mem

    return wrapper


# --------------------------------------------------
# Main worker (single GPU / single process)
# --------------------------------------------------

def main_worker(local_rank: int, args):
    """Entry point for each spawned process in DDP or single‑GPU inference."""

    # ---------------- Distributed setup ----------------
    if args.distributed:
        args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = dist.get_world_size()
        args.is_master = local_rank == 0
    else:
        args.local_rank = 0
        args.world_size = 1
        args.is_master = True
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # cuDNN speedup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # ---------------- Sanity checks ----------------
    assert os.path.isdir(args.data_root), f"[Error] Data root '{args.data_root}' does not exist."

    # ---------------- Build manager + load ckpt ----------------
    exp_manager = DLManager(args)
    exp_manager.load(args.checkpoint_path)

    # Ensure model on correct device (DLManager should handle this, but be safe)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    exp_manager.model.to(device)

    # ---------------- FLOPs & Param count ----------------
    if args.compute_metrics and args.is_master:
        flops, params = compute_flops_params(exp_manager.model, args.dummy_input_shape, device)
        if flops is not None:
            print(f"[Metrics] FLOPs/MACs per forward: {flops}\n[Metrics] #Parameters: {params}")

    # ---------------- Runtime & Memory ----------------
    wrapped_test = measure_runtime_memory(exp_manager.test)
    _, runtime_ms, peak_mem = wrapped_test()

    if args.is_master:
        print("[Metrics] Total inference runtime: {:.3f} s".format(runtime_ms / 1000))
        if peak_mem is not None:
            print("[Metrics] Peak GPU memory: {:.1f} MB".format(peak_mem))

    # ---------------- Cleanup ----------------
    if args.distributed:
        dist.destroy_process_group()


# --------------------------------------------------
# Argument parsing & launch
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference with metrics computation")

    # Paths / basic params
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (pth/pth.tar)")
    parser.add_argument("--save_root", type=str, required=True, help="Where to save inference outputs & logs")

    # Performance tuning
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Distributed flag
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel for multi‑GPU inference")

    # Metric flags
    parser.add_argument("--compute_metrics", action="store_true", help="Compute FLOPs/Params & measure runtime & memory")
    parser.add_argument(
        "--dummy_input_shape",
        type=int,
        nargs=3,
        default=[3, 224, 224],
        metavar=("C", "H", "W"),
        help="Dummy input shape (C H W) for FLOPs/params estimation if no real sample is used",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.distributed:
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 1, "[Error] --distributed requires at least 2 visible GPUs."
        mp.spawn(main_worker, nprocs=n_gpus, args=(args,))
    else:
        main_worker(0, args)
