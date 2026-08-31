# Physical Inductive Biases for Transferable Protein Binding Prediction

Reference implementation for the method described in *Physical Inductive Biases for
Transferable Protein Binding Prediction*.

The model augments an invariant point-attention (IPA) encoder with a **screened-Coulomb
electrostatic attention stream** and pretrains it to regress Poisson–Boltzmann interface
potentials on a large corpus of protein–protein interfaces, without affinity labels. Every
reported claim is made against a **size-matched, geometry-only ablation** trained on the same
folds, from the same pretrained initialisation, with the same seed and step budget, so the two
arms differ only in the presence of the electrostatic stream.

> **Scope.** This repository ships the **model and the training/evaluation trunk** only.
> Public benchmark datasets, the electrostatic self-supervised pretraining code, the
> Poisson–Boltzmann (APBS) target-generation pipeline, trained checkpoints, and the paper's
> downstream analysis scripts are **not** part of this code-only release; they are distributed
> separately (see **Data**, **Pretrained encoder**, and **Checkpoints**). Nothing here is
> redistributed from third-party datasets.

## Layout

```
code/                       shared model / data library
  model.py                  geometry-only IPA ΔΔG network (the size-matched control)
  module.py                 IPA blocks, encoders, shared layers
  dataset.py, skempi.py     dataset construction and SKEMPI parsing
  trainer.py, utils.py      training loop and helpers
  common_utils/             protein parsing / transforms (required at runtime)
  lea_experiment/
    run_ablation.py         train + evaluate one architecture arm (main entry point)
    lea_module_{c,d,g,r,x}.py   the electrostatic model and its matched control variants
configs/                    model configurations (config_63M.json is the one used throughout)
```

## Architectures

`python code/lea_experiment/run_ablation.py --arch <name>` trains and evaluates one arm
(3 folds, fixed step budget). All variants are matched in parameter count so that no difference
between them is attributable to capacity.

| `--arch` | electrostatic term | module | purpose |
|---|---|---|---|
| `base`  | none | `model.py` (`DDG_RDE_Network`) | size-matched geometry-only control |
| `leac`  | `q_i q_j exp(-κr/10) / (ε r)` | `lea_module_c.py` | the electrostatic model |
| `leac_gated` | screened Coulomb + learnable gate `g` | `lea_module_g.py` | quantifies that the stream is used (`sigmoid(g)`) |
| `leag`  | distance-bucketed per-head global bias, **no charge** | `lea_module_g.py` | does *any* global pairwise bias suffice, or the Coulomb form specifically? |
| `lear`  | `q_i q_j f(r)`, `f` learned | `lea_module_r.py` | is the *Coulomb shape* what matters? |
| `leard` | `f(r)`, `f` learned, no scalars | `lea_module_r.py` | would any pairwise radial channel do? |
| `leap`  | scalars permuted across residues | `lea_module_x.py` | does per-residue assignment matter? |
| `leaa`  | `\|q_i q_j\| exp(-κr/10) / (ε r)` | `lea_module_x.py` | does charge sign matter? |
| `leau`  | `q_i q_j / (ε r)` | `lea_module_x.py` | does screening matter? |

## Usage

Each run trains 3 folds at a fixed step budget and evaluates SKEMPI (three modes), ATLAS, and
the HER2 set, appending one row to `results/ablation/summary.tsv`.

```bash
# from-scratch training — runs with no external checkpoint (electrostatic model vs. control)
python code/lea_experiment/run_ablation.py --arch leac --pretrain no --tag leac
python code/lea_experiment/run_ablation.py --arch base --pretrain no --tag geo

# to reproduce the reported numbers, initialise from the pretrained electrostatic encoder:
python code/lea_experiment/run_ablation.py --arch leac --pretrain yes \
    --enc /path/to/es_enc_leac_50k_63M.ckpt --tag leac
```

`--pretrain yes` needs a pretrained encoder checkpoint: pass it with `--enc`, or place it at
`checkpoints/es_enc_leac_50k_63M.ckpt` (see **Pretrained encoder**). Datasets are read from
`data/` under the repo root; paths auto-resolve relative to that root, overridable with the
`LEAC_ROOT` environment variable. Add `--device cpu` to run without a GPU. `config_63M.json` is
the configuration used for the reported results.

## Data

All benchmarks are public and are **not** redistributed here: SKEMPI v2, AB-Bind, ATLAS, the
HER2/trastuzumab affinity-maturation set, the antibody escape panels, and the PPIRef interfaces
used for pretraining. Place them under `data/<benchmark>/` (see the path construction near the
top of `run_ablation.py`). Poisson–Boltzmann targets were generated with APBS and pdb2pqr
(AMBER force field).

## Pretrained encoder & checkpoints

The electrostatic self-supervised pretraining (Poisson–Boltzmann interface regression) produces
the encoder that `--pretrain yes` loads. That pretraining code and the trained encoder/fine-tuned
checkpoints are archived separately and are not part of this code-only repository. `--pretrain no`
reproduces the full pipeline from random initialisation without any external checkpoint.


## Requirements

See `requirements.txt`. Core stack: PyTorch ≥ 2.0, NumPy, pandas, SciPy, BioPython, easydict.

## License

MIT — see `LICENSE`.

## Citation

To be added on publication.
