"""统一消融 train+eval:一个 arch/pretrain 组合训 3 折(SKEMPI,固定步数)→ 评 SKEMPI三模式+ATLAS87+HER2。
用法(路径全自适配 flagship 目录,或用 LEAC_ROOT 覆盖):
    python run_ablation.py --arch leac --pretrain no  --tag leac_nossl
    python run_ablation.py --arch base --pretrain no  --tag ipa_nossl
    python run_ablation.py --arch leag --pretrain yes --tag leag
    python run_ablation.py --arch leac_gated --pretrain yes --tag leac_gate   # 报告 gate g
结果追加到 <root>/results/ablation/summary.tsv,ckpt 存 <root>/results/ablation/<tag>/。
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LEAC_ROOT", os.path.dirname(os.path.dirname(HERE)))
CODE = os.path.join(ROOT, "code"); os.environ.setdefault("LEAC_CODE", CODE)
sys.path.insert(0, CODE); sys.path.insert(0, HERE)
from model import DDG_RDE_Network
from lea_module_c import LEAC_DDG_Network
from lea_module_g import LEAG_DDG_Network, LEACGated_DDG_Network
from lea_module_d import LEAD_DDG_Network
# B1: functional-form controls for the screened-Coulomb stream
from lea_module_r import LEAR_DDG_Network, LEAD_RADIAL_DDG_Network
from lea_module_x import LEAP_DDG_Network, LEAA_DDG_Network, LEAU_DDG_Network
from skempi import SkempiDataset
from dataset import PaddingCollate, inf_iterator
from trainer import recursive_to
from utils import set_seed, eval_skempi_three_modes

NETS = {"base": DDG_RDE_Network, "leac": LEAC_DDG_Network, "leag": LEAG_DDG_Network, "leac_gated": LEACGated_DDG_Network, "leacd": LEAD_DDG_Network,
        "lear": LEAR_DDG_Network, "leard": LEAD_RADIAL_DDG_Network,
        "leap": LEAP_DDG_Network, "leaa": LEAA_DDG_Network, "leau": LEAU_DDG_Network}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(NETS), required=True)
    ap.add_argument("--pretrain", choices=["yes", "no"], required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None)
    # SKEMPI source. 'default' keeps the legacy behaviour byte-for-byte: cache-only, so
    # skempi.py finds no CSV, no fold_k columns, and falls back to its hash split
    # (FOLD_SEED or 3745754758) -- the partition every current result was produced on.
    # 'uniair' points at the UniAIR release, whose CSV carries explicit fold_0/1/2
    # columns; skempi.py prefers those, so this trains on UniAIR's OFFICIAL folds and
    # its deduplicated curation (5820 entries vs 6706 here; same 340 PDBs, same 5814
    # unique (PDB, mutation) pairs -- the difference is 892 duplicate measurements).
    # Needed because the two partitions are independent (31.2% agreement ~ chance) and
    # published methods score systematically LOWER on UniAIR's, so numbers are not
    # transferable across them.
    ap.add_argument("--skempi", choices=["default", "uniair"], default="default")
    # Model scale and pretrained encoder. Defaults reproduce the 63M flagship exactly, so
    # every previous command keeps its meaning; the 12M path exists for cheap architecture
    # screening before committing 63M-scale compute.
    ap.add_argument("--config", default="config_63M.json",
                    help="file under configs/, e.g. config_12M.json")
    ap.add_argument("--enc", default=None,
                    help="pretrained encoder .ckpt; default = the 63M electrostatic encoder")
    # The physical constants (kappa, eps, lambda_d, r0) are a handful of scalars whose
    # gradients are scaled by the charge amplitude, which is small (RMS|q| ~ 0.03). At the
    # 63M/40k-step budget they do move (lambda_D 14.43 -> 8.5-13.3 A, eps 1.693 -> 1.74-2.02),
    # but a 12M/2k-step screen left every one of them at its initial value to three decimals,
    # so that screen measured an untrained physics term and its conclusions were void.
    # These two options exist to make short screens meaningful. Defaults are no-ops, so the
    # flagship protocol is bit-for-bit unchanged.
    ap.add_argument("--exclude_complexes", default="",
                    help="path to a CSV with a `pdb` column; those complexes are dropped from "
                         "the TRAINING split only, never from validation. Empty by default, in "
                         "which case every code path below is byte-for-byte the previous one. "
                         "Used for the immune-excluded design, where all antibody-antigen and "
                         "TCR-pMHC complexes are removed from mutation supervision so that "
                         "transfer into those systems is measured rather than assumed.")
    ap.add_argument("--phys_lr_mult", type=float, default=1.0,
                    help="LR multiplier for kappa/eps/lambda_d/r0 (1.0 = unchanged)")
    ap.add_argument("--phys_init", action="store_true",
                    help="initialise lambda_D~11A and eps~1.88, near where 63M converged, "
                         "instead of 14.43A/1.693")
    # Validation-based early stopping ("heartbeat"). OFF by default -> flagship protocol is
    # byte-for-byte unchanged. When on, an inner-validation set is carved from the TRAINING
    # split only (never the reported SKEMPI fold-val, HER2 or ATLAS); every --eval_every steps
    # the inner-val overall-Pearson is measured, the best-so-far model state is kept, and if
    # --patience steps pass with no improvement the fold stops and its BEST-inner-val model is
    # used for the reported eval and saved. Selection therefore never peeks at any reported
    # metric, so it is standard model selection rather than test-set checkpoint cherry-picking.
    ap.add_argument("--early_stop", action="store_true",
                    help="enable inner-val early stopping / best-checkpoint selection")
    ap.add_argument("--patience", type=int, default=5000,
                    help="stop a fold after this many steps with no inner-val improvement")
    ap.add_argument("--eval_every", type=int, default=1000,
                    help="inner-val evaluation cadence (steps)")
    ap.add_argument("--inner_val_frac", type=float, default=0.1,
                    help="fraction of the TRAIN split held out as inner-val for early stopping")
    ap.add_argument("--es_select", choices=["inner", "foldval"], default="inner",
                    help="early-stop selection set. 'inner' (default) = 10%% held out of TRAIN, "
                         "leakage-free. 'foldval' = the fold's FULL val split; keeps all training data "
                         "but the val split IS the reported SKEMPI test, so the SKEMPI metric becomes "
                         "selection-biased (HER2/ATLAS remain independent).")
    ap.add_argument("--es_metric", choices=["overall", "perstructure"], default="overall",
                    help="early-stop selection metric on the selection set: pooled 'overall' Pearson "
                         "(default) or 'perstructure' Pearson (mean over complexes with >=10 muts, the "
                         "reported SKEMPI headline metric).")
    ap.add_argument("--warmstart_dir", default="",
                    help="if set, each fold initialises from <dir>/ddg_fold{fold}_step*.ckpt (warm-start "
                         "/ continue training) instead of the pretrained encoder.")
    ap.add_argument("--save_every_ckpt", action="store_true",
                    help="archive a checkpoint at every eval_every step (not just the selected best); "
                         "lets any step be re-evaluated offline. Costs ~250MB per saved step.")
    ap.add_argument("--device", default=None,
                    help="torch device; default = cuda if available else cpu.")
    a = ap.parse_args()
    args = argparse.Namespace(**json.loads(open(os.path.join(ROOT, "configs", a.config)).read()))
    if a.seed is not None: args.seed = a.seed
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed); coll = PaddingCollate()
    print(f"[seed] using seed={args.seed}", flush=True)
    D = lambda *p: os.path.join(ROOT, "data", *p)
    if a.skempi == "uniair":
        U = os.environ.get("SKEMPI_UNIAIR_DIR", D("skempi_uniair"))
        SK = (os.path.join(U, "skempi_ddg.csv"), os.path.join(U, "PDBs"), os.path.join(U, "cache"))
        assert os.path.exists(SK[0]), f"UniAIR SKEMPI CSV not found: {SK[0]}"
        print(f"[skempi] UniAIR release + OFFICIAL fold_0/1/2 columns: {SK[0]}", flush=True)
    else:
        SK = ("", "", D("SKEMPI_v2_cache"))          # cache-only: csv/pdb 不需要
        print("[skempi] legacy cache-only source, hash split (seed 3745754758)", flush=True)
    AT   = (D("atlas87","atlas_ddg.csv"), D("atlas87","PDBs"), D("atlas87","cache"))
    HER2 = (D("her2","her2_ddg.csv"),     D("her2","PDBs"),    D("her2","cache"))
    Net = NETS[a.arch]
    enc_path = a.enc or os.path.join(ROOT, "checkpoints", "es_enc_leac_50k_63M.ckpt")
    enc = torch.load(enc_path, map_location=dev) if a.pretrain == "yes" else None
    if enc is not None:
        print(f"[enc] {enc_path}", flush=True)
    outdir = os.path.join(ROOT, "results", "ablation", a.tag); os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, "DONE")):
        print(f"[{a.tag}] 已完成(DONE 标记),跳过", flush=True); return
    resume_path = os.path.join(outdir, "resume_state.pt")

    at_ds  = SkempiDataset(*AT,  num_cvfolds=3, cvfold_index=0, split="all", patch_size=args.patch_size)
    her2_ds= SkempiDataset(*HER2,num_cvfolds=3, cvfold_index=0, split="all", patch_size=args.patch_size)

    @torch.no_grad()
    def predict(m, ds):
        ps, ts = [], []
        for b in DataLoader(ds, batch_size=16, collate_fn=coll, num_workers=2):
            b = recursive_to(b, dev); _, o = m(b, None)
            ps += o['ddG_pred'].cpu().tolist(); ts += o['ddG_true'].cpu().tolist()
        return np.array(ps), np.array(ts)

    sk_rows = []; at_preds = []; her2_preds = []; at_t = her2_t = None; gates_all = []
    start_fold, start_step, _rmodel, _ropt = 0, 1, None, None
    if os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=dev, weights_only=False)
        start_fold, start_step = rs['fold'], rs['step'] + 1
        sk_rows, at_preds, her2_preds = rs['sk_rows'], rs['at_preds'], rs['her2_preds']
        at_t, her2_t, gates_all = rs['at_t'], rs['her2_t'], rs['gates_all']
        _rmodel, _ropt = rs.get('model'), rs.get('opt')
        print(f"[resume] 从 fold{start_fold} step{start_step-1} 恢复(已完成 {len(at_preds)} 折)", flush=True)

    def _save_resume(fold, step, m_, opt_):
        tmp = resume_path + ".tmp"   # 原子写:先写 tmp 再 rename,防杀进程时半截损坏
        torch.save({'fold': fold, 'step': step,
                    'model': None if m_ is None else m_.state_dict(),
                    'opt': None if opt_ is None else opt_.state_dict(),
                    'sk_rows': sk_rows, 'at_preds': at_preds, 'her2_preds': her2_preds,
                    'at_t': at_t, 'her2_t': her2_t, 'gates_all': gates_all}, tmp)
        os.replace(tmp, resume_path)

    for fold in range(start_fold, 3):
        m = Net(args).to(dev)
        if fold == start_fold and _rmodel is not None:
            m.load_state_dict(_rmodel); print(f"[fold{fold}] 从断点恢复模型", flush=True)
        elif a.warmstart_dir:
            import glob as _glob
            _ws = _glob.glob(os.path.join(a.warmstart_dir, f"ddg_fold{fold}_step*.ckpt"))
            if _ws:
                m.load_state_dict(torch.load(_ws[0], map_location=dev))
                print(f"[fold{fold}] warm-start 载入 {os.path.basename(_ws[0])} (续训)", flush=True)
            elif enc is not None:
                m.load_state_dict(enc, strict=False); print(f"[fold{fold}] warmstart缺该折ckpt,回退编码器", flush=True)
        elif enc is not None:
            miss = m.load_state_dict(enc, strict=False)
            print(f"[fold{fold}] 载入静电编码器: 命中{len([k for k in enc if k in m.state_dict()])}键", flush=True)
        else:
            print(f"[fold{fold}] 随机初始化(no pretrain)", flush=True)
        tr = SkempiDataset(*SK, num_cvfolds=3, cvfold_index=fold, split="train", patch_size=args.patch_size)
        if a.exclude_complexes:
            import pandas as _pd
            drop = {str(x).upper() for x in _pd.read_csv(a.exclude_complexes)["pdb"]}
            before = len(tr.entries)
            tr.entries = [e for e in tr.entries if e["pdbcode"].upper() not in drop]
            print(f"  [exclude] fold{fold} train {before} -> {len(tr.entries)} entries "
                  f"({before - len(tr.entries)} dropped, {len(drop)} complexes listed)",
                  flush=True)
        # selection set for early stopping
        iv_ds = None
        if a.early_stop and a.es_select == "inner":
            # hold out a fraction of the TRAIN split (leakage-free; never fold-val/HER2/ATLAS)
            import random as _rnd
            _rng = _rnd.Random(int(args.seed) * 100 + fold)
            _idx = list(range(len(tr.entries))); _rng.shuffle(_idx)
            _nv = max(1, int(len(_idx) * a.inner_val_frac))
            _val_ids = set(_idx[:_nv])
            _iv_entries = [tr.entries[i] for i in sorted(_val_ids)]
            tr.entries = [tr.entries[i] for i in _idx[_nv:]]
            iv_ds = SkempiDataset(*SK, num_cvfolds=3, cvfold_index=fold, split="train", patch_size=args.patch_size)
            iv_ds.entries = _iv_entries
            print(f"  [earlystop:inner] fold{fold} inner-val={len(_iv_entries)} train={len(tr.entries)} "
                  f"(patience={a.patience}, eval_every={a.eval_every})", flush=True)
        elif a.early_stop and a.es_select == "foldval":
            # full training data kept; select on the fold's FULL val split.
            # WARNING: this val split is the reported SKEMPI test -> SKEMPI metric is selection-biased.
            iv_ds = SkempiDataset(*SK, num_cvfolds=3, cvfold_index=fold, split="val", patch_size=args.patch_size)
            print(f"  [earlystop:foldval] fold{fold} select-on FOLD-VAL n={len(iv_ds.entries)} "
                  f"train(full)={len(tr.entries)} (patience={a.patience}, eval_every={a.eval_every}) "
                  f"[WARN: SKEMPI metric selection-biased for this run]", flush=True)
        dl = inf_iterator(DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=coll, num_workers=4))
        PHYS = ("log_kappa", "log_eps", "log_lambda_d", "log_r0")
        if a.phys_init:
            # start near the 63M converged region rather than at softplus(0)
            with torch.no_grad():
                for n_, p_ in m.named_parameters():
                    if n_.endswith("log_kappa"): p_.fill_(0.3934)   # -> lambda_D ~ 11 A
                    elif n_.endswith("log_eps"): p_.fill_(0.3442)   # -> eps ~ 1.88
            if fold == 0:
                print("[phys] init lambda_D~11A, eps~1.88", flush=True)
        phys = [p_ for n_, p_ in m.named_parameters() if n_.endswith(PHYS)]
        rest = [p_ for n_, p_ in m.named_parameters() if not n_.endswith(PHYS)]
        base_lr = args.lr * 0.3
        if a.phys_lr_mult != 1.0 and phys:
            opt = torch.optim.Adam([{"params": rest, "lr": base_lr},
                                    {"params": phys, "lr": base_lr * a.phys_lr_mult}])
            if fold == 0:
                print(f"[phys] {len(phys)} scalars at {a.phys_lr_mult}x LR", flush=True)
        else:
            opt = torch.optim.Adam(m.parameters(), lr=base_lr)
        if fold == start_fold and _ropt is not None: opt.load_state_dict(_ropt)
        m.train()
        f_start = start_step if fold == start_fold else 1
        _best_p, _best_step, _best_state = -1e9, 0, None
        def _inner_val_pearson():
            m.eval(); ps, ts, cs = [], [], []
            with torch.no_grad():
                for b in DataLoader(iv_ds, batch_size=32, collate_fn=coll, num_workers=2):
                    b = recursive_to(b, dev); _, o = m(b, None)
                    ps += o['ddG_pred'].cpu().tolist(); ts += o['ddG_true'].cpu().tolist(); cs += list(b['complex'])
            m.train()
            from scipy.stats import pearsonr as _pr
            if a.es_metric == "perstructure":
                from collections import defaultdict
                g = defaultdict(list)
                for c, t, p in zip(cs, ts, ps): g[c].append((t, p))
                rs = []
                for c, v in g.items():
                    if len(v) >= 10:
                        tt = [x[0] for x in v]; pp = [x[1] for x in v]
                        if len(set(tt)) > 1 and len(set(pp)) > 1:   # both must vary or pearsonr is nan
                            r = _pr(pp, tt)[0]
                            if r == r: rs.append(r)                 # drop any residual nan
                return float(sum(rs) / len(rs)) if rs else 0.0
            return _pr(ps, ts)[0] if len(set(ts)) > 1 else 0.0
        for e in range(f_start, a.steps + 1):
            b = recursive_to(next(dl), dev); loss, _ = m(b, None); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0); opt.step(); opt.zero_grad()
            if e % 2000 == 1: print(f"  [ft] 折{fold} 步{e}/{a.steps} loss={loss.item():.4f}", flush=True)
            if e % 1000 == 0: _save_resume(fold, e, m, opt)   # 每千步存断点(原子写)
            if a.save_every_ckpt and e % a.eval_every == 0:
                torch.save(m.state_dict(), os.path.join(outdir, f"ddg_fold{fold}_step{e}.ckpt"))
            if a.early_stop and e % a.eval_every == 0:
                _vp = _inner_val_pearson()
                if _vp > _best_p:
                    _best_p, _best_step = _vp, e
                    _best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                print(f"  [earlystop] 折{fold} 步{e} inner-val P={_vp:.4f} (best {_best_p:.4f}@{_best_step})", flush=True)
                if e - _best_step >= a.patience:
                    print(f"  [earlystop] 折{fold} 停于步{e};采用最优 ckpt @步{_best_step} (inner-val P={_best_p:.4f})", flush=True)
                    break
        if a.early_stop and _best_state is not None:
            m.load_state_dict(_best_state); es_best_step = _best_step
        else:
            es_best_step = a.steps
        m.eval()
        if a.arch == "leac_gated":
            g = [float(torch.sigmoid(blk.gate)) for blk in m.attn_encoder.blocks]
            gates_all.append(g); print(f"  折{fold} 静电流门控 sigmoid(g) 各层: {[round(x,4) for x in g]}", flush=True)
        va = SkempiDataset(*SK, num_cvfolds=3, cvfold_index=fold, split="val", patch_size=args.patch_size)
        with torch.no_grad():
            for b in DataLoader(va, batch_size=32, collate_fn=coll, num_workers=2):
                b = recursive_to(b, dev); _, o = m(b, None)
                for c, nm, yt, yp in zip(b['complex'], b['num_muts'], o['ddG_true'], o['ddG_pred']):
                    sk_rows.append({'complex': c, 'num_muts': int(nm), 'ddG': yt.item(), 'ddG_pred': yp.item()})
        ap_, at_ = predict(m, at_ds); hp_, ht_ = predict(m, her2_ds)
        at_preds.append(ap_); her2_preds.append(hp_); at_t, her2_t = at_, ht_
        torch.save(m.state_dict(), os.path.join(outdir, f"ddg_fold{fold}_step{es_best_step}.ckpt"))
        _save_resume(fold + 1, 0, None, None)   # 折完成:下一折从头(不存 model)
        start_step = 1; _rmodel = _ropt = None
        del m; torch.cuda.empty_cache()

    df = pd.DataFrame(sk_rows).assign(method='X')
    r = eval_skempi_three_modes(df).to_numpy()[0]  # 'all' row: [method,ovP,ovS,rmse,mae,auroc,psP,psS,mode]
    at_ens = np.mean(np.stack(at_preds), 0); her2_ens = np.mean(np.stack(her2_preds), 0)
    ps_p, ov_p, rmse, mae, auroc = float(r[6]), float(r[1]), float(r[3]), float(r[4]), float(r[5])
    at_p = pearsonr(at_ens, at_t)[0]; her2_p = pearsonr(her2_ens, her2_t)[0]; her2_s = spearmanr(her2_ens, her2_t)[0]
    print(f"\n===== [{a.tag}] arch={a.arch} pretrain={a.pretrain} =====", flush=True)
    print(f"[SKEMPI-val] PerStruct P={ps_p:.4f} | Overall P={ov_p:.4f} RMSE={rmse:.3f} AUROC={auroc:.4f}", flush=True)
    print(f"[ATLAS87集成] P={at_p:.4f}", flush=True)
    print(f"[HER2集成]   P={her2_p:.4f} S={her2_s:.4f}", flush=True)
    if gates_all:
        gm = np.mean(gates_all); print(f"[gate] 全折全层 sigmoid(g) 均值 = {gm:.4f}", flush=True)
    # 追加汇总
    summ = os.path.join(ROOT, "results", "ablation", "summary.tsv")
    head = not os.path.exists(summ)
    with open(summ, "a") as f:
        if head: f.write("tag\tarch\tpretrain\tseed\tSKEMPI_PS_P\tSKEMPI_Ov_P\tSKEMPI_RMSE\tSKEMPI_AUROC\tATLAS87_P\tHER2_P\tHER2_S\tgate_mean\n")
        gv = f"{np.mean(gates_all):.4f}" if gates_all else "NA"
        f.write(f"{a.tag}\t{a.arch}\t{a.pretrain}\t{args.seed}\t{ps_p:.4f}\t{ov_p:.4f}\t{rmse:.3f}\t{auroc:.4f}\t{at_p:.4f}\t{her2_p:.4f}\t{her2_s:.4f}\t{gv}\n")
    print(f"[summary] 追加到 {summ}", flush=True)
    if os.path.exists(resume_path): os.remove(resume_path)   # 完成:清断点
    open(os.path.join(outdir, "DONE"), "w").write("done")     # 打完成标记(重跑时跳过)
    print("ABLATION_RUN_DONE", flush=True)

if __name__ == "__main__":
    main()
