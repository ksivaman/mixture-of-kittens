import csv
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

from tests.utils import check_correctness


WARMUP_ITERS = 500
TIMED_ITERS = 100
PROFILE_WARMUP_ITERS = int(os.environ.get("PROFILE_WARMUP_ITERS", 5))
PROFILE_ITERS = max(1, int(os.environ.get("PROFILE_ITERS", 3)))


def get_num_local_experts(num_experts, world_size):
    if num_experts % world_size:
        raise ValueError(f"{num_experts} experts cannot be evenly divided across EP={world_size}")
    return num_experts // world_size


def get_tflops(latency_ms, num_local_tokens, topk, hidden_dim, intermediate_dim, backward=False):
    flops = 6 * (num_local_tokens + num_local_tokens * topk) * hidden_dim * intermediate_dim
    if backward:
        flops *= 2
    return flops / 1e9 / latency_ms


def check_benchmark_correctness(name, run_fwd, run_bwd, reference, tolerance, rank):
    output, context = run_fwd()
    backward = run_bwd(context)
    comparisons = (
        ("output", reference[0], output),
        ("d_x", reference[1], backward[0]),
        ("d_router_weights", reference[2], backward[1]),
        ("d_w_routed_gate", reference[3], backward[2]),
        ("d_w_routed_up", reference[4], backward[3]),
        ("d_w_routed_down", reference[5], backward[4]),
        ("d_w_shared_gate", reference[6], backward[5]),
        ("d_w_shared_up", reference[7], backward[6]),
        ("d_w_shared_down", reference[8], backward[7]),
    )
    for result_name, expected, result in comparisons:
        check_correctness(f"{name}/{result_name}", expected, result, tolerance, print_stats=rank == 0)


def bind_process_to_local_cpus(local_rank, local_world_size):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < local_world_size:
        return
    first = len(cpus) * local_rank // local_world_size
    last = len(cpus) * (local_rank + 1) // local_world_size
    os.sched_setaffinity(0, cpus[first:last])


def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("MOK_BENCHMARK_NUMA_BINDING", "1") != "0":
        bind_process_to_local_cpus(local_rank, int(os.environ["LOCAL_WORLD_SIZE"]))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    os.environ.setdefault("NCCL_IB_MERGE_NICS", "0")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)
    return rank, world_size, device


def median_rank_max_latency(samples, device):
    samples = torch.tensor(samples, dtype=torch.float64, device=device)
    rank_samples = [torch.empty_like(samples) for _ in range(dist.get_world_size())]
    dist.all_gather(rank_samples, samples)
    return torch.quantile(torch.stack(rank_samples).max(dim=0).values, 0.5).item()


def benchmark_fwd(run_fwd, device):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(TIMED_ITERS)]
    for _ in range(WARMUP_ITERS):
        output = run_fwd()
        output = None

    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    for start, end in events:
        start.record()
        output = run_fwd()
        end.record()
        output = None

    torch.cuda.synchronize()
    dist.barrier()
    return median_rank_max_latency([start.elapsed_time(end) for start, end in events], device)


def benchmark_bwd(run_fwd, run_bwd, device):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(TIMED_ITERS)]
    for _ in range(WARMUP_ITERS):
        context = run_fwd()[1]
        output = run_bwd(context)
        context = output = None

    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    for start, end in events:
        context = run_fwd()[1]
        start.record()
        output = run_bwd(context)
        end.record()
        context = output = None

    torch.cuda.synchronize()
    dist.barrier()
    return median_rank_max_latency([start.elapsed_time(end) for start, end in events], device)


def _profiler_event_value(event, generic_name, cuda_name):
    value = getattr(event, generic_name, None)
    if value is None:
        value = getattr(event, cuda_name, 0.0)
    return float(value or 0.0)


def _write_profiler_events(profiler, path):
    rows = []
    for event in profiler.events():
        count = int(getattr(event, "count", 1) or 1)
        self_cpu_time = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        cpu_time = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        self_gpu_time = _profiler_event_value(
            event,
            "self_device_time_total",
            "self_cuda_time_total",
        )
        gpu_time = _profiler_event_value(
            event,
            "device_time_total",
            "cuda_time_total",
        )
        rows.append(
            (
                str(getattr(event, "name", getattr(event, "key", ""))),
                str(getattr(event, "device_type", "")),
                count,
                self_cpu_time,
                cpu_time,
                cpu_time / count,
                self_gpu_time,
                gpu_time,
                gpu_time / count,
                int(getattr(event, "self_cpu_memory_usage", 0) or 0),
                int(getattr(event, "cpu_memory_usage", 0) or 0),
                int(
                    getattr(
                        event,
                        "self_device_memory_usage",
                        getattr(event, "self_cuda_memory_usage", 0),
                    )
                    or 0
                ),
                int(
                    getattr(
                        event,
                        "device_memory_usage",
                        getattr(event, "cuda_memory_usage", 0),
                    )
                    or 0
                ),
                repr(getattr(event, "input_shapes", "")),
            )
        )

    rows.sort(key=lambda row: (row[7], row[4]), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(
            (
                "name",
                "device_type",
                "calls",
                "self_cpu_time_us",
                "cpu_time_total_us",
                "cpu_time_avg_us",
                "self_gpu_time_us",
                "gpu_time_total_us",
                "gpu_time_avg_us",
                "self_cpu_memory_bytes",
                "cpu_memory_bytes",
                "self_gpu_memory_bytes",
                "gpu_memory_bytes",
                "input_shapes",
            )
        )
        writer.writerows(rows)


def profile_benchmark(name, run_fwd, run_bwd, output_dir, rank):
    """Profile forward/backward without using the CUDA-event timing helpers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(PROFILE_WARMUP_ITERS):
        output, context = run_fwd()
        backward = run_bwd(context)
        output = context = backward = None

    torch.cuda.synchronize()
    dist.barrier()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(PROFILE_ITERS):
            with torch.profiler.record_function(f"{name}/forward"):
                output, context = run_fwd()
            with torch.profiler.record_function(f"{name}/backward"):
                backward = run_bwd(context)
            output = context = backward = None
            profiler.step()

    torch.cuda.synchronize()
    dist.barrier()

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    prefix = output_dir / f"{slug}.rank{rank}"
    trace_path = Path(f"{prefix}.trace.json")
    table_path = Path(f"{prefix}.table.txt")
    events_path = Path(f"{prefix}.events.tsv")

    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=-1,
        max_name_column_width=1000,
    )
    table_path.write_text(f"{table}\n", encoding="utf-8")
    _write_profiler_events(profiler, events_path)

    if rank == 0:
        print(f"{name} profiler results:\n{table}")
        print(f"Chrome trace: {trace_path.resolve()}")
        print(f"Profiler table: {table_path.resolve()}")
        print(f"Raw event table: {events_path.resolve()}")
