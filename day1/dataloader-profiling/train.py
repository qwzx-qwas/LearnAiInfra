import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, Dataset


class SyntheticImageDataset(Dataset):
    def __init__(self, samples: int, num_classes: int, cpu_work: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randint(
            0,
            256,
            # 样本数 x 通道数 x 高度 x 宽度
            (samples, 3, 64, 64),
            dtype=torch.uint8,
            generator=generator,
        )
        self.labels = torch.randint(
            0,
            num_classes,
            (samples,),
            dtype=torch.long,
            generator=generator,
        )
        self.cpu_work = cpu_work

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int):
        image = self.images[index].float().div(255.0)

        # Simulate light CPU-side decode/augmentation work.
        for _ in range(self.cpu_work):
            image = image.mul(0.9).add(0.1)

        return image, self.labels[index]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch DataLoader performance with different num_workers."
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cpu-work", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = SyntheticImageDataset(
        samples=args.samples,
        num_classes=args.num_classes,
        cpu_work=args.cpu_work,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init_fn if args.num_workers > 0 else None,
        "generator": generator,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def build_profiler(device: torch.device, trace_path: Path):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=2, active=5, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prof=None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if prof is not None:
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return elapsed, total_loss / total_examples


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu" and torch.backends.mkldnn.is_available():
        # Avoid CPU Conv2d crashes seen with some PyTorch/oneDNN/VM combinations.
        torch.backends.mkldnn.enabled = False
    args.output.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args, device)
    model = SmallCNN(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trace_path = args.output / "trace.json"
    epoch_results = []

    for epoch in range(args.epochs):
        prof_context = (
            build_profiler(device, trace_path)
            if args.profile and epoch == 0
            else nullcontext()
        )
        with prof_context as prof:
            elapsed, loss = train_epoch(model, loader, criterion, optimizer, device, prof)

        epoch_result = {
            "epoch": epoch,
            "seconds": elapsed,
            "loss": loss,
            "kind": "cold" if epoch == 0 else "steady",
        }
        epoch_results.append(epoch_result)
        print(
            f"epoch={epoch} kind={epoch_result['kind']} "
            f"time={elapsed:.3f}s loss={loss:.4f}"
        )

    steady_epochs = [item["seconds"] for item in epoch_results[1:]]
    result = {
        "config": {
            "device": str(device),
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "samples": args.samples,
            "num_classes": args.num_classes,
            "cpu_work": args.cpu_work,
            "seed": args.seed,
            "profile": args.profile,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
        "epochs": epoch_results,
        "cold_epoch_seconds": epoch_results[0]["seconds"] if epoch_results else None,
        "steady_epoch_average_seconds": (
            sum(steady_epochs) / len(steady_epochs) if steady_epochs else None
        ),
    }

    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {result_path}")
    if args.profile:
        print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
