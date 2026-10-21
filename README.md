# IONA

IONA predicts mutation-induced changes in protein binding affinity using an invariant
point-attention encoder and a signed screened-Coulomb attention channel. The reference
configuration uses 1024-dimensional residue features, 256-dimensional pair features and
eight encoder blocks.

## Project layout

```text
iona/
  networks.py                 binding predictor and representation codebook
  encoders.py                 residue, pair and geometric encoders
  mutation_data.py            benchmark parsing and mutation datasets
  batching.py                 padded batches and data managers
  device_io.py                device transfer and cross-validation utilities
  runtime.py                  metrics, configuration and runtime utilities
  checkpoint_io.py            checkpoint key conversion
  primitives/                 protein geometry and data transforms
  experiments/
    run_benchmarks.py          training and evaluation entry point
    screened_coulomb.py       IONA screened-Coulomb model
    attention_controls.py     distance-bias and gated controls
    radial_controls.py        learned-radial and distance-only controls
    charge_controls.py        permutation, sign and screening controls
    solvation_control.py      additional solvation channel
configs/                      IONA model configurations
tests/                        namespace and numerical regression checks
```

Project-owned classes use the `IONA` prefix, and project-owned functions, variables,
model attributes and configuration fields use `iona_`. Python protocols, PyTorch hooks,
third-party APIs, amino-acid/atom symbols and cached benchmark field names retain their
required interfaces. Python source files contain no comments or docstrings.

## Installation

Use Python 3.10 or later and install the dependencies:

```bash
pip install -r requirements.txt
```

Choose a PyTorch build appropriate for your CPU or CUDA environment. The compatibility
checks for this naming revision used Python 3.12, CPU PyTorch and BioPython 1.81.

## Model variants

| `--arch` | Predictor | Attention channel |
|---|---|---|
| `iona` | `IONAPredictor` | signed screened-Coulomb |
| `iona_geometry` | `IONABindingPredictor` | geometric backbone |
| `iona_gated` | `IONAGatedPredictor` | gated screened-Coulomb |
| `iona_distance_bias` | `IONADistanceBiasPredictor` | learned distance buckets |
| `iona_learned_radial` | `IONALearnedRadialPredictor` | charge product and learned radial profile |
| `iona_distance_only` | `IONADistanceOnlyPredictor` | learned radial profile |
| `iona_permuted_charge` | `IONAPermutedChargePredictor` | permuted residue scalars |
| `iona_unsigned_charge` | `IONAUnsignedChargePredictor` | absolute scalar products |
| `iona_unscreened` | `IONAUnscreenedPredictor` | unscreened inverse distance |
| `iona_solvation` | `IONASolvationPredictor` | screened-Coulomb and solvation |

The screened-Coulomb channel adds a charge projection and two scalar parameters per
block. The variants retain their existing architectures and numerical behavior.

## Training and evaluation

Run commands from the repository root:

```bash
python -m iona.experiments.run_benchmarks --help
python -m iona.experiments.run_benchmarks --arch iona --pretrain no --tag iona_random
python -m iona.experiments.run_benchmarks --arch iona_geometry --pretrain no --tag iona_geometry_random
python -m iona.experiments.run_benchmarks --arch iona --pretrain yes --enc /path/to/encoder.ckpt --tag iona_pretrained
```

The direct script entry point also works:

```bash
python iona/experiments/run_benchmarks.py --arch iona --pretrain no --tag iona_random
```

The default configuration is `configs/iona_63m.json`. Pass `--config iona_12m.json`
or another filename under `configs/` to select a different configuration. Configuration
field names now carry the `iona_` prefix. Use `--device cpu` for CPU execution.
`IONA_ROOT` overrides the project data/configuration root; `IONA_CODE` identifies the
source directory for the import bootstrap.

Each run trains three folds and evaluates SKEMPI, ATLAS87 and HER2. The default budget
is 40,000 steps per fold. Results are appended to `results/ablation/summary.tsv`, and
checkpoints are saved under `results/ablation/<tag>/iona_fold<fold>_step<step>.ckpt`.

## Data and pretrained models

This repository contains the model and supervised training/evaluation code. Benchmark
data, prepared caches, APBS target generation, electrostatic self-supervised pretraining,
trained weights and downstream analysis scripts are supplied separately and are not
included here. Training with `--pretrain no` starts the supervised model from random
initialization; it still requires prepared benchmark data and does not run the
electrostatic self-supervised stage.

The default data locations are:

```text
data/SKEMPI_v2_cache/          entries.pkl and structures.pkl
data/atlas87/                 atlas_ddg.csv, PDBs/, cache/
data/her2/                    her2_ddg.csv, PDBs/, cache/
```

The default SKEMPI mode reads the prepared cache and uses the existing split seed
3745754758. `--skempi uniair` reads `data/skempi_uniair/skempi_ddg.csv`, `PDBs/` and
`cache/`, using explicit fold columns where available; `SKEMPI_UNIAIR_DIR` overrides
that location. The default pretrained encoder path is
`checkpoints/iona_encoder_50k_63m.ckpt`, or supply an existing file with `--enc`.

## Existing checkpoints and data

`load_state_dict` accepts both the previous model-attribute keys and the new IONA keys.
Conversion preserves tensor values and validates the resulting state dictionary using
the requested strictness. Newly saved weights use IONA keys. Cached dataset field names
and benchmark CSV columns retain their existing formats.

For continuation from an older run, rename its `ddg_fold<fold>_step<step>.ckpt` files to
`iona_fold<fold>_step<step>.ckpt` before using `--warmstart_dir`. Optimizer parameter order
and the numerical training logic are unchanged. Use a new output tag for the renamed
entry point.

## Validation

```bash
python -m unittest discover -s tests -v
```

The naming revision was also checked directly against commit `5d6e703`: all ten model
variants produced identical initialized tensors, evaluation predictions, losses and
parameter gradients on the same synthetic inputs. These checks validate the refactor;
they do not replace full benchmark reproduction with the original data and weights.

## License

MIT; see `LICENSE`.

## Citation

To be added on publication.
