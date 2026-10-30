# birds-of-a-feather-resnet

**SimCLR contrastive learning.** SimCLR on CIFAR-10: do semantic categories emerge in a model trained with no labels at all?

**Status:** not started.

## Question

Do semantic categories emerge in a model trained with no labels at all?

## Plan

- **Architecture:** ResNet-18 from torchvision, adapted for 32×32 images (3×3 first convolution, no max-pool), plus a 2-layer MLP projection head, trained with the NT-Xent contrastive loss on pairs of augmented crops.
- **Data:** CIFAR-10. Labels are used only for evaluation.

## Experiments

- Linear-probe and k-nearest-neighbour accuracy.
- A t-SNE or UMAP plot coloured by true class.
- Ablations of individual augmentations (colour jitter, cropping).
- Temperature and batch-size effects.

## Done when

- Classes form visible clusters, and linear-probe accuracy is far above a randomly initialized encoder.

Plus the general checklist:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
birds-of-a-feather-resnet/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/birds_of_a_feather_resnet/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Chen et al. 2020, SimCLR

## License

MIT
