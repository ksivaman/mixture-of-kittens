"""Benchmark expert parallelism with Transformer Engine's fused MXFP8 grouped MLP."""

# The grouped-MLP fusion is registered while Transformer Engine is imported.
import argparse
import os

os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"

from dataclasses import dataclass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-te-ep",
        action="store_true",
        help="Use Transformer Engine expert parallelism instead of DeepEP.",
    )
    return parser.parse_args()


# DeepEP must run its NCCL runtime check before PyTorch or Transformer Engine
# loads NCCL plugins. Keep it optional for the Transformer Engine EP mode.
_ARGS = parse_args() if __name__ == "__main__" else None
if _ARGS is None or not _ARGS.use_te_ep:
    import deep_ep

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import autocast, quantized_model_init
from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
)
from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU, Sequential
from transformer_engine.pytorch.ops.fused import GroupedMLP_CuTeGEMMGLU
from transformer_engine.pytorch.permutation import moe_permute_and_pad_with_probs, moe_unpermute
from transformer_engine.pytorch.utils import deinterleave_glu_tensor, interleave_glu_tensor

from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from tests.utils import MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
MXFP8_COMM_SMS = int(os.environ.get("MXFP8_COMM_SMS", 32))
# The cuDNN grouped GLU kernel requires the total token dimension and every
# expert offset to be aligned to its 256-row M tile.
MXFP8_ALIGNMENT = int(os.environ.get("MXFP8_ALIGNMENT", 256))
if MXFP8_ALIGNMENT % 256:
    raise ValueError(
        "MXFP8_ALIGNMENT must be a multiple of 256 for TE's fused grouped MLP, "
        f"got {MXFP8_ALIGNMENT}"
    )
GLU_INTERLEAVE_SIZE = 32
ENABLE_TORCH_COMPILE = False


@dataclass(frozen=True)
class DeepEpForwardContext:
    recv_x: torch.Tensor
    dense_probs: torch.Tensor
    safe_idx: torch.Tensor
    valid: torch.Tensor
    compact_output: torch.Tensor
    shared_output: torch.Tensor
    handle: object


@dataclass(frozen=True)
class TeEpForwardContext:
    routed_output: torch.Tensor
    shared_output: torch.Tensor


class TransformerEngineFusedBenchmark:
    def __init__(self, inputs, use_te_ep):
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
        self.num_sms = MXFP8_COMM_SMS
        self.num_local_experts = w_routed_gate.shape[0]
        self.num_experts = self.num_local_experts * dist.get_world_size()
        self.intermediate_dim = w_routed_gate.shape[1]
        self.use_te_ep = use_te_ep
        if use_te_ep:
            ep_bootstrap(
                dist.group.WORLD,
                num_experts=self.num_experts,
                max_tokens_per_rank=self.x.shape[0],
                hidden_dim=self.x.shape[1],
                num_topk=self.topk_experts.shape[1],
                # Eager mode sizes each receive buffer to the aligned routing
                # result, matching DeepEP's do_cpu_sync=True behavior below.
                recv_capacity_per_rank=None,
                max_num_sms=self.num_sms,
            )
            self.buffer = EpBuffer(
                top_k=self.topk_experts.shape[1],
                max_tokens_per_rank=self.x.shape[0],
                hidden_dim=self.x.shape[1],
                num_local_experts=self.num_local_experts,
                recv_capacity_per_rank=None,
                alignment=MXFP8_ALIGNMENT,
                device=self.x.device,
            )
        else:
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

        if not GroupedMLP_CuTeGEMMGLU.is_supported():
            raise RuntimeError(
                "Transformer Engine's fused grouped MLP is unavailable. It requires "
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP=1, an SM100 GPU, and supported cuDNN "
                "frontend grouped-GEMM kernels."
            )

        self.mxfp8_recipe = MXFP8BlockScaling()
        with quantized_model_init(enabled=True, recipe=self.mxfp8_recipe):
            self.fc1 = GroupedLinear(
                num_groups=self.num_local_experts,
                in_features=self.x.shape[1],
                out_features=2 * self.intermediate_dim,
                bias=False,
                dtype=torch.bfloat16,
                device=self.x.device,
                single_grouped_weight=True,
            )
            self.fc2 = GroupedLinear(
                num_groups=self.num_local_experts,
                in_features=self.intermediate_dim,
                out_features=self.x.shape[1],
                bias=False,
                dtype=torch.bfloat16,
                device=self.x.device,
                single_grouped_weight=True,
            )
            self.grouped_mlp = Sequential(
                self.fc1,
                ScaledSwiGLU(glu_interleave_size=GLU_INTERLEAVE_SIZE),
                self.fc2,
            )

        with torch.no_grad():
            fc1_weights = self.fc1.weight.quantized_tensors
            if fc1_weights is None:
                fc1_weights = self.fc1.weight.split_into_quantized_tensors()
            fc2_weights = self.fc2.weight.quantized_tensors
            if fc2_weights is None:
                fc2_weights = self.fc2.weight.split_into_quantized_tensors()
            for expert_idx in range(self.num_local_experts):
                fc1_weight = torch.cat(
                    (w_routed_gate[expert_idx], w_routed_up[expert_idx]),
                    dim=0,
                )
                fc1_weight = interleave_glu_tensor(fc1_weight, GLU_INTERLEAVE_SIZE)
                fc1_weights[expert_idx].copy_(fc1_weight)
                fc2_weights[expert_idx].copy_(
                    w_routed_down[expert_idx]
                )

        self.routed_parameters = tuple(self.fc1.parameters()) + tuple(self.fc2.parameters())
        self.x_routed = self.x.detach().requires_grad_()
        self.router_weights_routed = self.router_weights.detach().requires_grad_()
        self.x_shared = self.x.detach().requires_grad_()
        self.w_shared_gate = w_shared_gate.detach().requires_grad_()
        self.w_shared_up = w_shared_up.detach().requires_grad_()
        self.w_shared_down = w_shared_down.detach().requires_grad_()

    def run_grouped_mlp(self, x, m_splits, probs):
        with autocast(enabled=True, recipe=self.mxfp8_recipe):
            # Each fusible op receives its extra tensor inputs in sequence:
            # FC1 splits, activation scales, then FC2 splits.
            return self.grouped_mlp(x, m_splits, probs, m_splits)

    @torch.compiler.disable
    def run_grouped_mlp_eager(self, x, m_splits, probs):
        return self.run_grouped_mlp(x, m_splits, probs)

    def run_fwd_deepep(self):
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
        routing_map = torch.zeros(
            (recv_x.shape[0], self.num_local_experts),
            dtype=torch.int32,
            device=recv_x.device,
        )
        routing_map.scatter_add_(1, safe_idx, valid.to(torch.int32))
        dense_probs = torch.zeros(
            (recv_x.shape[0], self.num_local_experts),
            dtype=torch.float32,
            device=recv_x.device,
        )
        dense_probs.scatter_add_(1, safe_idx, recv_weights * valid)
        dense_probs.requires_grad_()
        tokens_per_expert = torch.tensor(
            handle.num_recv_tokens_per_expert_list,
            dtype=torch.int64,
            device=recv_x.device,
        )

        expert_x, expert_probs, row_map, pad_offsets, m_splits = (
            moe_permute_and_pad_with_probs(
                recv_x,
                dense_probs,
                routing_map,
                tokens_per_expert,
                MXFP8_ALIGNMENT,
            )
        )
        if ENABLE_TORCH_COMPILE:
            expert_output = self.run_grouped_mlp_eager(expert_x, m_splits, expert_probs)
        else:
            expert_output = self.run_grouped_mlp(expert_x, m_splits, expert_probs)
        # ScaledSwiGLU already applies the routing probabilities, so unpermute
        # only accumulates the top-k expert outputs.
        compact_output = moe_unpermute(
            expert_output,
            row_map,
            restore_shape=recv_x.shape,
            map_type="mask",
            pad_offsets=pad_offsets,
        )
        routed_output, _, event = self.buffer.combine(
            compact_output,
            handle=handle,
            async_with_compute_stream=True,
        )
        event.current_stream_wait()
        output = (routed_output.float() + shared_output.float()).to(torch.bfloat16)
        return output, DeepEpForwardContext(
            recv_x,
            dense_probs,
            safe_idx,
            valid,
            compact_output,
            shared_output,
            handle,
        )

    def run_fwd_te_ep(self):
        recv_x, recv_weights, tokens_per_expert = ep_dispatch(
            self.buffer,
            self.x_routed,
            self.topk_experts,
            self.router_weights_routed,
        )

        gate_shared = self.x_shared @ self.w_shared_gate.T
        up_shared = self.x_shared @ self.w_shared_up.T
        hidden_shared = F.silu(gate_shared).mul_(up_shared)
        shared_output = hidden_shared @ self.w_shared_down.T

        # TE EP returns expert-major rows and aligned split sizes, so no
        # additional permutation/padding pass is needed before the fused MLP.
        if ENABLE_TORCH_COMPILE:
            expert_output = self.run_grouped_mlp_eager(
                recv_x,
                tokens_per_expert,
                recv_weights,
            )
        else:
            expert_output = self.run_grouped_mlp(
                recv_x,
                tokens_per_expert,
                recv_weights,
            )
        routed_output = ep_combine(
            self.buffer,
            expert_output,
            num_local_tokens=self.x.shape[0],
        )
        output = (routed_output.float() + shared_output.float()).to(torch.bfloat16)
        return output, TeEpForwardContext(routed_output, shared_output)

    def run_fwd(self):
        if self.use_te_ep:
            return self.run_fwd_te_ep()
        return self.run_fwd_deepep()

    def run_bwd_deepep(self, context):
        d_compact_output, _, _, _, event = self.buffer.dispatch(
            self.d_output,
            handle=context.handle,
            num_sms=context.handle.num_sms,
            do_expand=False,
            async_with_compute_stream=True,
        )

        d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = (
            torch.autograd.grad(
                context.shared_output,
                (
                    self.x_shared,
                    self.w_shared_gate,
                    self.w_shared_up,
                    self.w_shared_down,
                ),
                self.d_output,
            )
        )

        event.current_stream_wait()
        # TE Sequential implements its joint fused backward through
        # _OperationFuserAutogradFunction; autograd dispatches directly to the
        # GroupedMLP_CuTeGEMMGLU.fuser_backward method selected below.
        d_recv_x, *d_routed_parameters, d_dense_probs = torch.autograd.grad(
            context.compact_output,
            (context.recv_x, *self.routed_parameters, context.dense_probs),
            d_compact_output,
        )
        d_recv_weights = (
            d_dense_probs.gather(1, context.safe_idx) * context.valid
        ).contiguous()
        d_x_routed, d_router_weights, event = self.buffer.combine(
            d_recv_x,
            topk_weights=d_recv_weights,
            handle=context.handle,
            async_with_compute_stream=True,
        )
        event.current_stream_wait()

        d_fc1, d_fc2 = d_routed_parameters
        d_x = (d_x_routed.float() + d_x_shared.float()).to(torch.bfloat16)
        return (
            d_x,
            d_router_weights,
            d_fc1,
            d_fc2,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )

    def run_bwd_te_ep(self, context):
        d_x_routed, d_router_weights, *d_routed_parameters = torch.autograd.grad(
            context.routed_output,
            (
                self.x_routed,
                self.router_weights_routed,
                *self.routed_parameters,
            ),
            self.d_output,
        )
        d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = (
            torch.autograd.grad(
                context.shared_output,
                (
                    self.x_shared,
                    self.w_shared_gate,
                    self.w_shared_up,
                    self.w_shared_down,
                ),
                self.d_output,
            )
        )

        d_fc1, d_fc2 = d_routed_parameters
        d_x = (d_x_routed.float() + d_x_shared.float()).to(torch.bfloat16)
        return (
            d_x,
            d_router_weights,
            d_fc1,
            d_fc2,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )

    def run_bwd(self, context):
        if self.use_te_ep:
            return self.run_bwd_te_ep(context)
        return self.run_bwd_deepep(context)

    def format_backward_for_correctness(self, backward):
        (
            d_x,
            d_router_weights,
            d_fc1,
            d_fc2,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        ) = backward
        d_fc1 = torch.stack(
            [
                deinterleave_glu_tensor(grad, GLU_INTERLEAVE_SIZE)
                for grad in d_fc1
            ]
        )
        d_gate, d_up = d_fc1.split(self.intermediate_dim, dim=1)
        return (
            d_x,
            d_router_weights,
            d_gate,
            d_up,
            d_fc2,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )

    def assert_fused_grouped_mlp(self):
        module_groups = self.grouped_mlp._module_groups
        if not module_groups:
            raise RuntimeError("Transformer Engine did not initialize the grouped MLP fuser")
        forward_ops = module_groups[0]._forward_ops
        backward_ops = module_groups[0]._backward_ops
        if (
            len(forward_ops) != 1
            or len(backward_ops) != 1
            or not isinstance(forward_ops[0][0], GroupedMLP_CuTeGEMMGLU)
            or backward_ops[0][0] is not forward_ops[0][0]
        ):
            raise RuntimeError(
                "Transformer Engine did not select the joint fused grouped MLP for "
                "both forward and backward"
            )

        fused_op = forward_ops[0][0]
        if fused_op.grouped_gemm_wgrad_kernel() is None:
            raise RuntimeError(
                "Transformer Engine selected its grouped-GEMM fallback for weight gradients"
            )

    def destroy(self):
        if self.use_te_ep:
            ep_finalize()
        else:
            self.buffer.destroy()


def main(args):
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(
        rank,
        device,
        NUM_EXPERTS,
        num_local_experts,
        TOPK,
        NUM_LOCAL_TOKENS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
    )

    if rank == 0:
        print(
            f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} "
            f"({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}"
        )
        ep_backend = "Transformer Engine EP" if args.use_te_ep else "DeepEP"
        print(
            f"{ep_backend}: MXFP8 comm SMs={MXFP8_COMM_SMS}, "
            f"MXFP8 alignment={MXFP8_ALIGNMENT}"
        )
        print(f"torch.compile={ENABLE_TORCH_COMPILE} (max-autotune-no-cudagraphs)")

    ep_backend = "Transformer Engine EP" if args.use_te_ep else "DeepEP"
    name = f"{ep_backend} + Transformer Engine fused MXFP8"
    benchmark = TransformerEngineFusedBenchmark(inputs, args.use_te_ep)
    run_fwd = benchmark.run_fwd
    run_bwd = benchmark.run_bwd
    if ENABLE_TORCH_COMPILE:
        run_fwd = torch.compile(run_fwd, mode="max-autotune-no-cudagraphs")
        run_bwd = torch.compile(run_bwd, mode="max-autotune-no-cudagraphs")
    reference = run_reference_bf16(*inputs)

    def run_bwd_for_correctness(context):
        return benchmark.format_backward_for_correctness(run_bwd(context))

    check_benchmark_correctness(
        name,
        run_fwd,
        run_bwd_for_correctness,
        reference,
        MXFP8_TOLERANCE,
        rank,
    )
    benchmark.assert_fused_grouped_mlp()
    del reference

    fwd_ms = benchmark_fwd(run_fwd, device)
    bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
    if rank == 0:
        fwd_tflops = get_tflops(
            fwd_ms,
            NUM_LOCAL_TOKENS,
            TOPK,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )
        bwd_tflops = get_tflops(
            bwd_ms,
            NUM_LOCAL_TOKENS,
            TOPK,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
            backward=True,
        )
        print(
            f"{name}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; "
            f"backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s"
        )
    benchmark.destroy()
    del benchmark

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(_ARGS)
