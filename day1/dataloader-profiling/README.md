# DataLoader Profiling Experiment

This experiment compares one variable: `DataLoader(num_workers=0)` versus
`DataLoader(num_workers=2)`.

## CPU smoke test

Install `py-spy` if you want a Python flamegraph:

```bash
pip install -r requirements.txt
```

```bash
python train.py --device cpu --num-workers 0 --epochs 2 --output outputs/cpu_workers0
python train.py --device cpu --num-workers 2 --epochs 2 --output outputs/cpu_workers2
```

## GPU experiment

```bash
python train.py \
  --device cuda \
  --num-workers 0 \
  --epochs 4 \
  --batch-size 128 \
  --profile \
  --output outputs/workers0

python train.py \
  --device cuda \
  --num-workers 2 \
  --epochs 4 \
  --batch-size 128 \
  --profile \
  --output outputs/workers2
```

Each run writes:

- `results.json`: epoch times, cold epoch time, steady epoch average
- `trace.json`: PyTorch profiler trace when `--profile` is enabled

Open `trace.json` in Chrome at `chrome://tracing`.
