import argparse
import os

import torch
import torch.distributed as dist
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-per-rank", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--reuse-buffer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.tokens % world_size:
        raise ValueError(f"tokens ({args.tokens}) must be divisible by world size ({world_size})")

    output = None
    for iteration in range(args.iterations):
        chunks = [
            torch.full(
                (args.tokens // world_size, args.hidden_per_rank),
                rank * 1000 + destination * 100 + iteration,
                dtype=torch.bfloat16,
                device=f"npu:{local_rank}",
            )
            for destination in range(world_size)
        ]
        send = torch.cat(chunks, dim=0).contiguous()
        if output is None or not args.reuse_buffer:
            output = torch.empty_like(send)

        print(
            f"before rank={rank} iteration={iteration} "
            f"shape={tuple(send.shape)} stride={send.stride()} "
            f"contiguous={send.is_contiguous()} stream={torch.npu.current_stream()}",
            flush=True,
        )
        dist.all_to_all_single(output, send)
        torch.npu.current_stream().synchronize()

        received = output.chunk(world_size, dim=0)
        for source, chunk in enumerate(received):
            expected = source * 1000 + rank * 100 + iteration
            if not torch.all(chunk == expected):
                raise AssertionError(
                    f"rank={rank} iteration={iteration} source={source} expected={expected}"
                )
        print(f"after rank={rank} iteration={iteration} verified=true", flush=True)

    dist.barrier()
    if rank == 0:
        print("PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
