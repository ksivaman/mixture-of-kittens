import argparse
import os

import torch.distributed as dist

from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed, profile_benchmark
from mok import functional, ops
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
BF16_FWD_COMM_SMS = int(os.environ.get("BF16_FWD_COMM_SMS", 24))
BF16_BWD_COMM_SMS = int(os.environ.get("BF16_BWD_COMM_SMS", 28))
MXFP8_FWD_COMM_SMS = int(os.environ.get("MXFP8_FWD_COMM_SMS", 36))
MXFP8_BWD_COMM_SMS = int(os.environ.get("MXFP8_BWD_COMM_SMS", 36))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 4096))
MACROBATCH_SIZE = int(os.environ.get("MACROBATCH_SIZE", 32 * MINIBATCH_SIZE))

BF16_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=BF16_FWD_COMM_SMS,
    bwd_num_comm_sms=BF16_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
)
MXFP8_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=MXFP8_FWD_COMM_SMS,
    bwd_num_comm_sms=MXFP8_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
)


class MoKBenchmark:
    def __init__(self, inputs):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
            self.d_output,
        ) = inputs
        self.num_local_experts = self.w_routed_gate.shape[0]
        self.workspace = functional.get_workspace(BF16_CONFIG, dist.group.WORLD, device=self.x.device, num_local_tokens=self.x.shape[0], hidden_size=self.x.shape[1], topk=self.topk_experts.shape[1])
        self.w_gate_mxfp8 = ops.mxfp8_quantize(self.w_routed_gate, True, True)
        self.w_up_mxfp8 = ops.mxfp8_quantize(self.w_routed_up, True, True)
        self.w_down_mxfp8 = ops.mxfp8_quantize(self.w_routed_down, True, True)

    def run_bf16_fwd(self):
        schedule = functional.build_schedule(self.workspace, BF16_CONFIG, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            BF16_CONFIG,
            self.workspace,
            schedule,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
        )
        return output, (schedule, forward_context)

    def run_bf16_bwd(self, context):
        schedule, forward_context = context
        return functional.backward(
            BF16_CONFIG,
            self.workspace,
            schedule,
            forward_context,
            self.d_output,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
        )

    def run_mxfp8_fwd(self):
        schedule = functional.build_schedule(self.workspace, MXFP8_CONFIG, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            MXFP8_CONFIG,
            self.workspace,
            schedule,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_gate_mxfp8[:2],
            self.w_up_mxfp8[:2],
            self.w_down_mxfp8[:2],
        )
        return output, (schedule, forward_context)

    def run_mxfp8_bwd(self, context):
        schedule, forward_context = context
        return functional.backward(
            MXFP8_CONFIG,
            self.workspace,
            schedule,
            forward_context,
            self.d_output,
            self.x,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_gate_mxfp8,
            self.w_up_mxfp8,
            self.w_down_mxfp8[2:],
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        nargs="?",
        const="profiles",
        metavar="OUTPUT_DIR",
        help="Use torch.profiler and write Chrome trace/table files (default: profiles).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    benchmark = MoKBenchmark(inputs)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}")
        print(f"BF16 comm SMs={BF16_FWD_COMM_SMS}/{BF16_BWD_COMM_SMS}, MXFP8 comm SMs={MXFP8_FWD_COMM_SMS}/{MXFP8_BWD_COMM_SMS}, minibatch={MINIBATCH_SIZE}, macrobatch={MACROBATCH_SIZE}")

    variants = (
        ("BF16", benchmark.run_bf16_fwd, benchmark.run_bf16_bwd, BF16_TOLERANCE),
        ("MXFP8", benchmark.run_mxfp8_fwd, benchmark.run_mxfp8_bwd, MXFP8_TOLERANCE),
    )
    reference = run_reference_bf16(*inputs)
    for precision, run_fwd, run_bwd, tolerance in variants:
        check_benchmark_correctness(precision, run_fwd, run_bwd, reference, tolerance, rank)
    del reference

    for precision, run_fwd, run_bwd, _ in variants:
        if args.profile is not None:
            profile_benchmark(
                f"MoK {precision}",
                run_fwd,
                run_bwd,
                args.profile,
                rank,
            )
            continue
        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{precision}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
