import os, sys, json, argparse, warnings

warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

iona_HERE = os.path.dirname(os.path.abspath(__file__))
iona_ROOT = os.environ.get("IONA_ROOT", os.path.dirname(os.path.dirname(iona_HERE)))
iona_CODE = os.path.join(iona_ROOT, "iona")
os.environ.setdefault("IONA_CODE", iona_CODE)
sys.path.insert(0, iona_ROOT)
sys.path.insert(0, iona_HERE)
from iona.networks import IONABindingPredictor
from iona.checkpoint_io import iona_normalize_state_dict
from iona.experiments.screened_coulomb import IONAPredictor
from iona.experiments.attention_controls import (
    IONADistanceBiasPredictor,
    IONAGatedPredictor,
)
from iona.experiments.solvation_control import IONASolvationPredictor
from iona.experiments.radial_controls import (
    IONALearnedRadialPredictor,
    IONADistanceOnlyPredictor,
)
from iona.experiments.charge_controls import (
    IONAPermutedChargePredictor,
    IONAUnsignedChargePredictor,
    IONAUnscreenedPredictor,
)
from iona.mutation_data import IONAMutationDataset
from iona.batching import IONAPaddedBatch, iona_inf_iterator
from iona.device_io import iona_move_to_device
from iona.runtime import iona_set_seed, iona_evaluate_mutation_subsets

iona_NETS = {
    "iona_geometry": IONABindingPredictor,
    "iona": IONAPredictor,
    "iona_distance_bias": IONADistanceBiasPredictor,
    "iona_gated": IONAGatedPredictor,
    "iona_solvation": IONASolvationPredictor,
    "iona_learned_radial": IONALearnedRadialPredictor,
    "iona_distance_only": IONADistanceOnlyPredictor,
    "iona_permuted_charge": IONAPermutedChargePredictor,
    "iona_unsigned_charge": IONAUnsignedChargePredictor,
    "iona_unscreened": IONAUnscreenedPredictor,
}


def iona_run_benchmarks():
    iona_ap = argparse.ArgumentParser()
    iona_ap.add_argument(
        "--arch", choices=list(iona_NETS), required=True, dest="iona_arch"
    )
    iona_ap.add_argument(
        "--pretrain", choices=["yes", "no"], required=True, dest="iona_pretrain"
    )
    iona_ap.add_argument("--steps", type=int, default=40000, dest="iona_steps")
    iona_ap.add_argument("--tag", required=True, dest="iona_tag")
    iona_ap.add_argument("--seed", type=int, default=None, dest="iona_seed")
    iona_ap.add_argument(
        "--skempi", choices=["default", "uniair"], default="default", dest="iona_skempi"
    )
    iona_ap.add_argument(
        "--config",
        default="iona_63m.json",
        help="file under configs/, e.g. iona_12m.json",
        dest="iona_config",
    )
    iona_ap.add_argument(
        "--enc",
        default=None,
        help="pretrained encoder .ckpt; default = the 63M electrostatic encoder",
        dest="iona_enc",
    )
    iona_ap.add_argument(
        "--exclude_complexes",
        default="",
        help="path to a CSV with a `pdb` column; those complexes are dropped from the TRAINING split only, never from validation. Empty by default, in which case every code path below is byte-for-byte the previous one. Used for the immune-excluded design, where all antibody-antigen and TCR-pMHC complexes are removed from mutation supervision so that transfer into those systems is measured rather than assumed.",
        dest="iona_exclude_complexes",
    )
    iona_ap.add_argument(
        "--phys_lr_mult",
        type=float,
        default=1.0,
        help="LR multiplier for kappa/eps/lambda_d/r0 (1.0 = unchanged)",
        dest="iona_phys_lr_mult",
    )
    iona_ap.add_argument(
        "--phys_init",
        action="store_true",
        help="initialise lambda_D~11A and eps~1.88, near where 63M converged, instead of 14.43A/1.693",
        dest="iona_phys_init",
    )
    iona_ap.add_argument(
        "--early_stop",
        action="store_true",
        help="enable inner-val early stopping / best-checkpoint selection",
        dest="iona_early_stop",
    )
    iona_ap.add_argument(
        "--patience",
        type=int,
        default=5000,
        help="stop a fold after this many steps with no inner-val improvement",
        dest="iona_patience",
    )
    iona_ap.add_argument(
        "--eval_every",
        type=int,
        default=1000,
        help="inner-val evaluation cadence (steps)",
        dest="iona_eval_every",
    )
    iona_ap.add_argument(
        "--inner_val_frac",
        type=float,
        default=0.1,
        help="fraction of the TRAIN split held out as inner-val for early stopping",
        dest="iona_inner_val_frac",
    )
    iona_ap.add_argument(
        "--es_select",
        choices=["inner", "foldval"],
        default="inner",
        help="early-stop selection set. 'inner' (default) = 10%% held out of TRAIN, leakage-free. 'foldval' = the fold's FULL val split; keeps all training data but the val split IS the reported SKEMPI test, so the SKEMPI metric becomes selection-biased (HER2/ATLAS remain independent).",
        dest="iona_es_select",
    )
    iona_ap.add_argument(
        "--es_metric",
        choices=["overall", "perstructure"],
        default="overall",
        help="early-stop selection metric on the selection set: pooled 'overall' Pearson (default) or 'perstructure' Pearson (mean over complexes with >=10 muts, the reported SKEMPI headline metric).",
        dest="iona_es_metric",
    )
    iona_ap.add_argument(
        "--warmstart_dir",
        default="",
        help="if set, each fold initialises from <dir>/iona_fold{fold}_step*.ckpt (warm-start / continue training) instead of the pretrained encoder.",
        dest="iona_warmstart_dir",
    )
    iona_ap.add_argument(
        "--save_every_ckpt",
        action="store_true",
        help="archive a checkpoint at every eval_every step (not just the selected best); lets any step be re-evaluated offline. Costs ~250MB per saved step.",
        dest="iona_save_every_ckpt",
    )
    iona_ap.add_argument(
        "--device",
        default=None,
        help="torch device; default = cuda if available else cpu.",
        dest="iona_device",
    )
    iona_a = iona_ap.parse_args()
    iona_args = argparse.Namespace(
        **json.loads(
            open(os.path.join(iona_ROOT, "configs", iona_a.iona_config)).read()
        )
    )
    if iona_a.iona_seed is not None:
        iona_args.iona_seed = iona_a.iona_seed
    iona_dev = iona_a.iona_device or ("cuda" if torch.cuda.is_available() else "cpu")
    iona_set_seed(iona_args.iona_seed)
    iona_coll = IONAPaddedBatch()
    print(f"[seed] using seed={iona_args.iona_seed}", flush=True)
    iona_D = lambda *iona_p: os.path.join(iona_ROOT, "data", *iona_p)
    if iona_a.iona_skempi == "uniair":
        iona_U = os.environ.get("SKEMPI_UNIAIR_DIR", iona_D("skempi_uniair"))
        iona_SK = (
            os.path.join(iona_U, "skempi_ddg.csv"),
            os.path.join(iona_U, "PDBs"),
            os.path.join(iona_U, "cache"),
        )
        assert os.path.exists(iona_SK[0]), f"UniAIR SKEMPI CSV not found: {iona_SK[0]}"
        print(
            f"[skempi] UniAIR release + OFFICIAL fold_0/1/2 columns: {iona_SK[0]}",
            flush=True,
        )
    else:
        iona_SK = ("", "", iona_D("SKEMPI_v2_cache"))
        print(
            "[skempi] legacy cache-only source, hash split (seed 3745754758)",
            flush=True,
        )
    iona_AT = (
        iona_D("atlas87", "atlas_ddg.csv"),
        iona_D("atlas87", "PDBs"),
        iona_D("atlas87", "cache"),
    )
    iona_HER2 = (
        iona_D("her2", "her2_ddg.csv"),
        iona_D("her2", "PDBs"),
        iona_D("her2", "cache"),
    )
    iona_Net = iona_NETS[iona_a.iona_arch]
    iona_enc_path = iona_a.iona_enc or os.path.join(
        iona_ROOT, "checkpoints", "iona_encoder_50k_63m.ckpt"
    )
    iona_enc = (
        torch.load(iona_enc_path, map_location=iona_dev)
        if iona_a.iona_pretrain == "yes"
        else None
    )
    if iona_enc is not None:
        iona_enc = iona_normalize_state_dict(iona_enc)
        print(f"[enc] {iona_enc_path}", flush=True)
    iona_outdir = os.path.join(iona_ROOT, "results", "ablation", iona_a.iona_tag)
    os.makedirs(iona_outdir, exist_ok=True)
    if os.path.exists(os.path.join(iona_outdir, "DONE")):
        print(f"[{iona_a.iona_tag}] 已完成(DONE 标记),跳过", flush=True)
        return
    iona_resume_path = os.path.join(iona_outdir, "resume_state.pt")
    iona_at_ds = IONAMutationDataset(
        *iona_AT,
        iona_num_cvfolds=3,
        iona_cvfold_index=0,
        iona_split="all",
        iona_patch_size=iona_args.iona_patch_size,
    )
    iona_her2_ds = IONAMutationDataset(
        *iona_HER2,
        iona_num_cvfolds=3,
        iona_cvfold_index=0,
        iona_split="all",
        iona_patch_size=iona_args.iona_patch_size,
    )

    @torch.no_grad()
    def iona_predict(iona_m, iona_ds):
        iona_ps, iona_ts = ([], [])
        for iona_b in DataLoader(
            iona_ds, batch_size=16, collate_fn=iona_coll, num_workers=2
        ):
            iona_b = iona_move_to_device(iona_b, iona_dev)
            _, iona_o = iona_m(iona_b, None)
            iona_ps += iona_o["ddG_pred"].cpu().tolist()
            iona_ts += iona_o["ddG_true"].cpu().tolist()
        return (np.array(iona_ps), np.array(iona_ts))

    iona_sk_rows = []
    iona_at_preds = []
    iona_her2_preds = []
    iona_at_t = iona_her2_t = None
    iona_gates_all = []
    iona_start_fold, iona_start_step, iona_rmodel, iona_ropt = (0, 1, None, None)
    if os.path.exists(iona_resume_path):
        iona_rs = torch.load(
            iona_resume_path, map_location=iona_dev, weights_only=False
        )
        iona_start_fold, iona_start_step = (iona_rs["fold"], iona_rs["step"] + 1)
        iona_sk_rows, iona_at_preds, iona_her2_preds = (
            iona_rs["sk_rows"],
            iona_rs["at_preds"],
            iona_rs["her2_preds"],
        )
        iona_at_t, iona_her2_t, iona_gates_all = (
            iona_rs["at_t"],
            iona_rs["her2_t"],
            iona_rs["gates_all"],
        )
        iona_rmodel, iona_ropt = (iona_rs.get("model"), iona_rs.get("opt"))
        print(
            f"[resume] 从 fold{iona_start_fold} step{iona_start_step - 1} 恢复(已完成 {len(iona_at_preds)} 折)",
            flush=True,
        )

    def iona_save_resume(iona_fold, iona_step, iona_m_, iona_opt_):
        iona_tmp = iona_resume_path + ".tmp"
        torch.save(
            {
                "fold": iona_fold,
                "step": iona_step,
                "model": None if iona_m_ is None else iona_m_.state_dict(),
                "opt": None if iona_opt_ is None else iona_opt_.state_dict(),
                "sk_rows": iona_sk_rows,
                "at_preds": iona_at_preds,
                "her2_preds": iona_her2_preds,
                "at_t": iona_at_t,
                "her2_t": iona_her2_t,
                "gates_all": iona_gates_all,
            },
            iona_tmp,
        )
        os.replace(iona_tmp, iona_resume_path)

    for iona_fold in range(iona_start_fold, 3):
        iona_m = iona_Net(iona_args).to(iona_dev)
        if iona_fold == iona_start_fold and iona_rmodel is not None:
            iona_m.load_state_dict(iona_rmodel)
            print(f"[fold{iona_fold}] 从断点恢复模型", flush=True)
        elif iona_a.iona_warmstart_dir:
            import glob as _glob

            iona_ws = _glob.glob(
                os.path.join(
                    iona_a.iona_warmstart_dir, f"iona_fold{iona_fold}_step*.ckpt"
                )
            )
            if iona_ws:
                iona_m.load_state_dict(torch.load(iona_ws[0], map_location=iona_dev))
                print(
                    f"[fold{iona_fold}] warm-start 载入 {os.path.basename(iona_ws[0])} (续训)",
                    flush=True,
                )
            elif iona_enc is not None:
                iona_m.load_state_dict(iona_enc, strict=False)
                print(f"[fold{iona_fold}] warmstart缺该折ckpt,回退编码器", flush=True)
        elif iona_enc is not None:
            iona_miss = iona_m.load_state_dict(iona_enc, strict=False)
            print(
                f"[fold{iona_fold}] 载入静电编码器: 命中{len([iona_k for iona_k in iona_enc if iona_k in iona_m.state_dict()])}键",
                flush=True,
            )
        else:
            print(f"[fold{iona_fold}] 随机初始化(no pretrain)", flush=True)
        iona_tr = IONAMutationDataset(
            *iona_SK,
            iona_num_cvfolds=3,
            iona_cvfold_index=iona_fold,
            iona_split="train",
            iona_patch_size=iona_args.iona_patch_size,
        )
        if iona_a.iona_exclude_complexes:
            import pandas as _pd

            iona_drop = {
                str(iona_x).upper()
                for iona_x in _pd.read_csv(iona_a.iona_exclude_complexes)["pdb"]
            }
            iona_before = len(iona_tr.iona_entries)
            iona_tr.iona_entries = [
                iona_e
                for iona_e in iona_tr.iona_entries
                if iona_e["pdbcode"].upper() not in iona_drop
            ]
            print(
                f"  [exclude] fold{iona_fold} train {iona_before} -> {len(iona_tr.iona_entries)} entries ({iona_before - len(iona_tr.iona_entries)} dropped, {len(iona_drop)} complexes listed)",
                flush=True,
            )
        iona_iv_ds = None
        if iona_a.iona_early_stop and iona_a.iona_es_select == "inner":
            import random as _rnd

            iona_rng = _rnd.Random(int(iona_args.iona_seed) * 100 + iona_fold)
            iona_idx = list(range(len(iona_tr.iona_entries)))
            iona_rng.shuffle(iona_idx)
            iona_nv = max(1, int(len(iona_idx) * iona_a.iona_inner_val_frac))
            iona_val_ids = set(iona_idx[:iona_nv])
            iona_iv_entries = [
                iona_tr.iona_entries[iona_i] for iona_i in sorted(iona_val_ids)
            ]
            iona_tr.iona_entries = [
                iona_tr.iona_entries[iona_i] for iona_i in iona_idx[iona_nv:]
            ]
            iona_iv_ds = IONAMutationDataset(
                *iona_SK,
                iona_num_cvfolds=3,
                iona_cvfold_index=iona_fold,
                iona_split="train",
                iona_patch_size=iona_args.iona_patch_size,
            )
            iona_iv_ds.iona_entries = iona_iv_entries
            print(
                f"  [earlystop:inner] fold{iona_fold} inner-val={len(iona_iv_entries)} train={len(iona_tr.iona_entries)} (patience={iona_a.iona_patience}, eval_every={iona_a.iona_eval_every})",
                flush=True,
            )
        elif iona_a.iona_early_stop and iona_a.iona_es_select == "foldval":
            iona_iv_ds = IONAMutationDataset(
                *iona_SK,
                iona_num_cvfolds=3,
                iona_cvfold_index=iona_fold,
                iona_split="val",
                iona_patch_size=iona_args.iona_patch_size,
            )
            print(
                f"  [earlystop:foldval] fold{iona_fold} select-on FOLD-VAL n={len(iona_iv_ds.iona_entries)} train(full)={len(iona_tr.iona_entries)} (patience={iona_a.iona_patience}, eval_every={iona_a.iona_eval_every}) [WARN: SKEMPI metric selection-biased for this run]",
                flush=True,
            )
        iona_dl = iona_inf_iterator(
            DataLoader(
                iona_tr,
                batch_size=iona_args.iona_batch_size,
                shuffle=True,
                collate_fn=iona_coll,
                num_workers=4,
            )
        )
        iona_PHYS = (
            "iona_log_kappa",
            "iona_log_eps",
            "iona_log_lambda_d",
            "iona_log_r0",
        )
        if iona_a.iona_phys_init:
            with torch.no_grad():
                for iona_n_, iona_p_ in iona_m.named_parameters():
                    if iona_n_.endswith("log_kappa"):
                        iona_p_.fill_(0.3934)
                    elif iona_n_.endswith("log_eps"):
                        iona_p_.fill_(0.3442)
            if iona_fold == 0:
                print("[phys] init lambda_D~11A, eps~1.88", flush=True)
        iona_phys = [
            iona_p_
            for iona_n_, iona_p_ in iona_m.named_parameters()
            if iona_n_.endswith(iona_PHYS)
        ]
        iona_rest = [
            iona_p_
            for iona_n_, iona_p_ in iona_m.named_parameters()
            if not iona_n_.endswith(iona_PHYS)
        ]
        iona_base_lr = iona_args.iona_lr * 0.3
        if iona_a.iona_phys_lr_mult != 1.0 and iona_phys:
            iona_opt = torch.optim.Adam(
                [
                    {"params": iona_rest, "lr": iona_base_lr},
                    {
                        "params": iona_phys,
                        "lr": iona_base_lr * iona_a.iona_phys_lr_mult,
                    },
                ]
            )
            if iona_fold == 0:
                print(
                    f"[phys] {len(iona_phys)} scalars at {iona_a.iona_phys_lr_mult}x LR",
                    flush=True,
                )
        else:
            iona_opt = torch.optim.Adam(iona_m.parameters(), lr=iona_base_lr)
        if iona_fold == iona_start_fold and iona_ropt is not None:
            iona_opt.load_state_dict(iona_ropt)
        iona_m.train()
        iona_f_start = iona_start_step if iona_fold == iona_start_fold else 1
        iona_best_p, iona_best_step, iona_best_state = (-1000000000.0, 0, None)

        def iona_inner_val_pearson():
            iona_m.eval()
            iona_ps, iona_ts, iona_cs = ([], [], [])
            with torch.no_grad():
                for iona_b in DataLoader(
                    iona_iv_ds, batch_size=32, collate_fn=iona_coll, num_workers=2
                ):
                    iona_b = iona_move_to_device(iona_b, iona_dev)
                    _, iona_o = iona_m(iona_b, None)
                    iona_ps += iona_o["ddG_pred"].cpu().tolist()
                    iona_ts += iona_o["ddG_true"].cpu().tolist()
                    iona_cs += list(iona_b["complex"])
            iona_m.train()
            from scipy.stats import pearsonr as _pr

            if iona_a.iona_es_metric == "perstructure":
                from collections import defaultdict

                iona_g = defaultdict(list)
                for iona_c, iona_t, iona_p in zip(iona_cs, iona_ts, iona_ps):
                    iona_g[iona_c].append((iona_t, iona_p))
                iona_rs = []
                for iona_c, iona_v in iona_g.items():
                    if len(iona_v) >= 10:
                        iona_tt = [iona_x[0] for iona_x in iona_v]
                        iona_pp = [iona_x[1] for iona_x in iona_v]
                        if len(set(iona_tt)) > 1 and len(set(iona_pp)) > 1:
                            iona_r = _pr(iona_pp, iona_tt)[0]
                            if iona_r == iona_r:
                                iona_rs.append(iona_r)
                return float(sum(iona_rs) / len(iona_rs)) if iona_rs else 0.0
            return _pr(iona_ps, iona_ts)[0] if len(set(iona_ts)) > 1 else 0.0

        for iona_e in range(iona_f_start, iona_a.iona_steps + 1):
            iona_b = iona_move_to_device(next(iona_dl), iona_dev)
            iona_loss, _ = iona_m(iona_b, None)
            iona_loss.backward()
            torch.nn.utils.clip_grad_norm_(iona_m.parameters(), 100.0)
            iona_opt.step()
            iona_opt.zero_grad()
            if iona_e % 2000 == 1:
                print(
                    f"  [ft] 折{iona_fold} 步{iona_e}/{iona_a.iona_steps} loss={iona_loss.item():.4f}",
                    flush=True,
                )
            if iona_e % 1000 == 0:
                iona_save_resume(iona_fold, iona_e, iona_m, iona_opt)
            if iona_a.iona_save_every_ckpt and iona_e % iona_a.iona_eval_every == 0:
                torch.save(
                    iona_m.state_dict(),
                    os.path.join(
                        iona_outdir, f"iona_fold{iona_fold}_step{iona_e}.ckpt"
                    ),
                )
            if iona_a.iona_early_stop and iona_e % iona_a.iona_eval_every == 0:
                iona_vp = iona_inner_val_pearson()
                if iona_vp > iona_best_p:
                    iona_best_p, iona_best_step = (iona_vp, iona_e)
                    iona_best_state = {
                        iona_k: iona_v.detach().cpu().clone()
                        for iona_k, iona_v in iona_m.state_dict().items()
                    }
                print(
                    f"  [earlystop] 折{iona_fold} 步{iona_e} inner-val P={iona_vp:.4f} (best {iona_best_p:.4f}@{iona_best_step})",
                    flush=True,
                )
                if iona_e - iona_best_step >= iona_a.iona_patience:
                    print(
                        f"  [earlystop] 折{iona_fold} 停于步{iona_e};采用最优 ckpt @步{iona_best_step} (inner-val P={iona_best_p:.4f})",
                        flush=True,
                    )
                    break
        if iona_a.iona_early_stop and iona_best_state is not None:
            iona_m.load_state_dict(iona_best_state)
            iona_es_best_step = iona_best_step
        else:
            iona_es_best_step = iona_a.iona_steps
        iona_m.eval()
        if iona_a.iona_arch == "iona_gated":
            iona_g = [
                float(torch.sigmoid(iona_blk.iona_gate))
                for iona_blk in iona_m.iona_attn_encoder.iona_blocks
            ]
            iona_gates_all.append(iona_g)
            print(
                f"  折{iona_fold} 静电流门控 sigmoid(g) 各层: {[round(iona_x, 4) for iona_x in iona_g]}",
                flush=True,
            )
        iona_va = IONAMutationDataset(
            *iona_SK,
            iona_num_cvfolds=3,
            iona_cvfold_index=iona_fold,
            iona_split="val",
            iona_patch_size=iona_args.iona_patch_size,
        )
        with torch.no_grad():
            for iona_b in DataLoader(
                iona_va, batch_size=32, collate_fn=iona_coll, num_workers=2
            ):
                iona_b = iona_move_to_device(iona_b, iona_dev)
                _, iona_o = iona_m(iona_b, None)
                for iona_c, iona_nm, iona_yt, iona_yp in zip(
                    iona_b["complex"],
                    iona_b["num_muts"],
                    iona_o["ddG_true"],
                    iona_o["ddG_pred"],
                ):
                    iona_sk_rows.append(
                        {
                            "complex": iona_c,
                            "num_muts": int(iona_nm),
                            "ddG": iona_yt.item(),
                            "ddG_pred": iona_yp.item(),
                        }
                    )
        iona_ap_, iona_at_ = iona_predict(iona_m, iona_at_ds)
        iona_hp_, iona_ht_ = iona_predict(iona_m, iona_her2_ds)
        iona_at_preds.append(iona_ap_)
        iona_her2_preds.append(iona_hp_)
        iona_at_t, iona_her2_t = (iona_at_, iona_ht_)
        torch.save(
            iona_m.state_dict(),
            os.path.join(
                iona_outdir, f"iona_fold{iona_fold}_step{iona_es_best_step}.ckpt"
            ),
        )
        iona_save_resume(iona_fold + 1, 0, None, None)
        iona_start_step = 1
        iona_rmodel = iona_ropt = None
        del iona_m
        torch.cuda.empty_cache()
    iona_df = pd.DataFrame(iona_sk_rows).assign(method="X")
    iona_r = iona_evaluate_mutation_subsets(iona_df).to_numpy()[0]
    iona_at_ens = np.mean(np.stack(iona_at_preds), 0)
    iona_her2_ens = np.mean(np.stack(iona_her2_preds), 0)
    iona_ps_p, iona_ov_p, iona_rmse, iona_mae, iona_auroc = (
        float(iona_r[6]),
        float(iona_r[1]),
        float(iona_r[3]),
        float(iona_r[4]),
        float(iona_r[5]),
    )
    iona_at_p = pearsonr(iona_at_ens, iona_at_t)[0]
    iona_her2_p = pearsonr(iona_her2_ens, iona_her2_t)[0]
    iona_her2_s = spearmanr(iona_her2_ens, iona_her2_t)[0]
    print(
        f"\n===== [{iona_a.iona_tag}] arch={iona_a.iona_arch} pretrain={iona_a.iona_pretrain} =====",
        flush=True,
    )
    print(
        f"[SKEMPI-val] PerStruct P={iona_ps_p:.4f} | Overall P={iona_ov_p:.4f} RMSE={iona_rmse:.3f} AUROC={iona_auroc:.4f}",
        flush=True,
    )
    print(f"[ATLAS87集成] P={iona_at_p:.4f}", flush=True)
    print(f"[HER2集成]   P={iona_her2_p:.4f} S={iona_her2_s:.4f}", flush=True)
    if iona_gates_all:
        iona_gm = np.mean(iona_gates_all)
        print(f"[gate] 全折全层 sigmoid(g) 均值 = {iona_gm:.4f}", flush=True)
    iona_summ = os.path.join(iona_ROOT, "results", "ablation", "summary.tsv")
    iona_head = not os.path.exists(iona_summ)
    with open(iona_summ, "a") as iona_f:
        if iona_head:
            iona_f.write(
                "tag\tarch\tpretrain\tseed\tSKEMPI_PS_P\tSKEMPI_Ov_P\tSKEMPI_RMSE\tSKEMPI_AUROC\tATLAS87_P\tHER2_P\tHER2_S\tgate_mean\n"
            )
        iona_gv = f"{np.mean(iona_gates_all):.4f}" if iona_gates_all else "NA"
        iona_f.write(
            f"{iona_a.iona_tag}\t{iona_a.iona_arch}\t{iona_a.iona_pretrain}\t{iona_args.iona_seed}\t{iona_ps_p:.4f}\t{iona_ov_p:.4f}\t{iona_rmse:.3f}\t{iona_auroc:.4f}\t{iona_at_p:.4f}\t{iona_her2_p:.4f}\t{iona_her2_s:.4f}\t{iona_gv}\n"
        )
    print(f"[summary] 追加到 {iona_summ}", flush=True)
    if os.path.exists(iona_resume_path):
        os.remove(iona_resume_path)
    open(os.path.join(iona_outdir, "DONE"), "w").write("done")
    print("IONA_RUN_DONE", flush=True)


if __name__ == "__main__":
    iona_run_benchmarks()
