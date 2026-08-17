"""Benchmarks used DeepEP 2.1.0+gitdd758caf, PyTorch 2.13.0+cu130, TorchAO 0.18.0+git55c8e6e3, Transformer Engine 2.17.0+git2e559f06, CUTLASS DSL 4.5.3, NCCL 2.30.7, cuBLAS 13.6.0.2, CUDA 13.0"""

from dataclasses import dataclass
import os

import deep_ep
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer_engine.pytorch.permutation import moe_permute, moe_permute_and_pad_with_probs, moe_unpermute

from benchmarks.impl_utils import grouped_mm, prequantize_mxfp8_weight
from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
BF16_COMM_SMS = int(os.environ.get("BF16_COMM_SMS", 32))
MXFP8_COMM_SMS = int(os.environ.get("MXFP8_COMM_SMS", 24))
MXFP8_ALIGNMENT = int(os.environ.get("MXFP8_ALIGNMENT", 256))
ENABLE_TORCH_COMPILE = True


@dataclass(frozen=True)
class ForwardContext:
    recv_x: torch.Tensor
    dense_probs: torch.Tensor
    safe_idx: torch.Tensor
    valid: torch.Tensor
    compact_output: torch.Tensor
    shared_output: torch.Tensor
    handle: object


class DeepEpTorchBenchmark:
    def __init__(self, inputs, mxfp8):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            self.d_output,
        ) = inputs
        self.mxfp8 = mxfp8
        self.num_sms = MXFP8_COMM_SMS if mxfp8 else BF16_COMM_SMS
        self.expert_alignment = MXFP8_ALIGNMENT if mxfp8 else 1
        self.num_local_experts = w_routed_gate.shape[0]
        self.num_experts = self.num_local_experts * dist.get_world_size()
        self.buffer = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=self.x.shape[0],
            hidden=self.x.shape[1],
            num_topk=self.topk_experts.shape[1],
            use_fp8_dispatch=False,
            deterministic=True,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
        )

        self.x_shared = self.x.detach().requires_grad_()
        self.w_gate_t = w_routed_gate.detach().transpose(1, 2).requires_grad_()
        self.w_up_t = w_routed_up.detach().transpose(1, 2).requires_grad_()
        self.w_down_t = w_routed_down.detach().transpose(1, 2).requires_grad_()
        self.w_gate_mxfp8 = prequantize_mxfp8_weight(w_routed_gate) if mxfp8 else None
        self.w_up_mxfp8 = prequantize_mxfp8_weight(w_routed_up) if mxfp8 else None
        self.w_down_mxfp8 = prequantize_mxfp8_weight(w_routed_down) if mxfp8 else None
        self.w_shared_gate = w_shared_gate.detach().requires_grad_()
        self.w_shared_up = w_shared_up.detach().requires_grad_()
        self.w_shared_down = w_shared_down.detach().requires_grad_()

    def run_fwd(self):
        recv_x, recv_idx, recv_weights, handle, event = self.buffer.dispatch(
            self.x,
            topk_idx=self.topk_experts,
            topk_weights=self.router_weights,
            num_experts=self.num_experts,
            expert_alignment=1,
            num_sms=self.num_sms,
            do_cpu_sync=True,
            do_expand=False,
            async_with_compute_stream=True,
        )
        self.num_sms = handle.num_sms

        gate_shared = self.x_shared @ self.w_shared_gate.T
        up_shared = self.x_shared @ self.w_shared_up.T
        hidden_shared = F.silu(gate_shared).mul_(up_shared)
        shared_output = hidden_shared @ self.w_shared_down.T

        event.current_stream_wait()
        recv_x.requires_grad_()
        valid = recv_idx >= 0
        safe_idx = recv_idx.clamp_min(0)
        routing_map = torch.zeros((recv_x.shape[0], self.num_local_experts), dtype=torch.int32, device=recv_x.device)
        routing_map.scatter_add_(1, safe_idx, valid.to(torch.int32))
        dense_probs = torch.zeros((recv_x.shape[0], self.num_local_experts), dtype=torch.float32, device=recv_x.device)
        dense_probs.scatter_add_(1, safe_idx, recv_weights * valid)
        dense_probs.requires_grad_()
        tokens_per_expert = torch.tensor(handle.num_recv_tokens_per_expert_list, dtype=torch.int64, device=recv_x.device)

        if self.mxfp8:
            expert_x, _, row_map, pad_offsets, expert_sizes = moe_permute_and_pad_with_probs(
                recv_x,
                dense_probs,
                routing_map,
                tokens_per_expert,
                self.expert_alignment,
            )
        else:
            expert_x, row_map = moe_permute(
                recv_x,
                routing_map,
                sum(handle.num_recv_tokens_per_expert_list),
                map_type="mask",
            )
            pad_offsets = None
            expert_sizes = tokens_per_expert
        expert_offsets = torch.cumsum(expert_sizes, 0, dtype=torch.int32)

        gate = grouped_mm(expert_x, self.w_gate_t, self.w_gate_mxfp8, expert_offsets)
        up = grouped_mm(expert_x, self.w_up_t, self.w_up_mxfp8, expert_offsets)
        hidden = F.silu(gate).mul_(up)
        expert_output = grouped_mm(hidden, self.w_down_t, self.w_down_mxfp8, expert_offsets)
        compact_output = moe_unpermute(
            expert_output,
            row_map,
            merging_probs=dense_probs,
            restore_shape=recv_x.shape,
            map_type="mask",
            pad_offsets=pad_offsets,
        )

        routed_output, _, event = self.buffer.combine(compact_output, handle=handle, async_with_compute_stream=True)
        event.current_stream_wait()
        output = (routed_output.float() + shared_output.float()).to(torch.bfloat16)
        return output, ForwardContext(recv_x, dense_probs, safe_idx, valid, compact_output, shared_output, handle)

    def run_bwd(self, context):
        d_compact_output, _, _, _, event = self.buffer.dispatch(
            self.d_output,
            handle=context.handle,
            num_sms=context.handle.num_sms,
            do_expand=False,
            async_with_compute_stream=True,
        )

        d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = torch.autograd.grad(
            context.shared_output,
            (self.x_shared, self.w_shared_gate, self.w_shared_up, self.w_shared_down),
            self.d_output,
        )

        event.current_stream_wait()
        d_recv_x, d_w_gate_t, d_w_up_t, d_w_down_t, d_dense_probs = torch.autograd.grad(
            context.compact_output,
            (context.recv_x, self.w_gate_t, self.w_up_t, self.w_down_t, context.dense_probs),
            d_compact_output,
        )
        d_recv_weights = (d_dense_probs.gather(1, context.safe_idx) * context.valid).contiguous()
        d_x_routed, d_router_weights, event = self.buffer.combine(
            d_recv_x,
            topk_weights=d_recv_weights,
            handle=context.handle,
            async_with_compute_stream=True,
        )
        event.current_stream_wait()
        d_x = (d_x_routed.float() + d_x_shared.float()).to(torch.bfloat16)
        return (
            d_x,
            d_router_weights,
            d_w_gate_t.transpose(1, 2),
            d_w_up_t.transpose(1, 2),
            d_w_down_t.transpose(1, 2),
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )

    def destroy(self):
        self.buffer.destroy()


def main():
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}")
        print(f"DeepEP: BF16 comm SMs={BF16_COMM_SMS}, MXFP8 comm SMs={MXFP8_COMM_SMS}")
        print(f"torch.compile={ENABLE_TORCH_COMPILE} (max-autotune-no-cudagraphs)")

    variants = (("DeepEP + PyTorch BF16", False, BF16_TOLERANCE), ("DeepEP + TorchAO MXFP8", True, MXFP8_TOLERANCE))
    for name, mxfp8, tolerance in variants:
        if mxfp8 and num_local_experts > 32:
            if rank == 0:
                print(f"{name} unsupported with {num_local_experts} local experts")
            continue

        benchmark = DeepEpTorchBenchmark(inputs, mxfp8)
        run_fwd = benchmark.run_fwd
        run_bwd = benchmark.run_bwd
        if ENABLE_TORCH_COMPILE:
            run_fwd = torch.compile(run_fwd, mode="max-autotune-no-cudagraphs")  # cuda graphs trigger repeated re-recording
            run_bwd = torch.compile(run_bwd, mode="max-autotune-no-cudagraphs")
        reference = run_reference_bf16(*inputs)
        check_benchmark_correctness(name, run_fwd, run_bwd, reference, tolerance, rank)
        del reference

        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{name}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")
        benchmark.destroy()
        del benchmark

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
