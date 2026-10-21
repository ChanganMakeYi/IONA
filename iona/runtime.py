import os
import yaml
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.linalg
from tqdm.auto import tqdm
from easydict import EasyDict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression


def iona_set_seed(iona_seed):
    torch.manual_seed(iona_seed)
    torch.cuda.manual_seed_all(iona_seed)
    np.random.seed(iona_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def iona_check_dir(iona_path, iona_overwrite=True):
    if not os.path.exists(iona_path):
        os.makedirs(iona_path)
    elif iona_overwrite:
        shutil.rmtree(iona_path)
        os.makedirs(iona_path)
    else:
        pass


def iona_save_code(iona_save_dir):
    iona_save_code_dir = os.path.join(iona_save_dir, "codes/")
    iona_check_dir(iona_save_code_dir + "primitives/modules/")
    iona_check_dir(iona_save_code_dir + "primitives/protein/")
    iona_check_dir(iona_save_code_dir + "primitives/transforms/")
    for iona_file in os.listdir("../iona"):
        if ".py" in iona_file:
            shutil.copyfile("../iona/" + iona_file, iona_save_code_dir + iona_file)
    for iona_file in os.listdir("../iona/primitives/modules/"):
        if ".py" in iona_file:
            shutil.copyfile(
                "../iona/primitives/modules/" + iona_file,
                iona_save_code_dir + "primitives/modules/" + iona_file,
            )
    for iona_file in os.listdir("../iona/primitives/protein/"):
        if ".py" in iona_file:
            shutil.copyfile(
                "../iona/primitives/protein/" + iona_file,
                iona_save_code_dir + "primitives/protein/" + iona_file,
            )
    for iona_file in os.listdir("../iona/primitives/transforms/"):
        if ".py" in iona_file:
            shutil.copyfile(
                "../iona/primitives/transforms/" + iona_file,
                iona_save_code_dir + "primitives/transforms/" + iona_file,
            )


class IONAConfigLoader(yaml.SafeLoader):

    def __init__(self, iona_stream):
        try:
            self.iona_root = os.path.split(iona_stream.name)[0]
        except AttributeError:
            self.iona_root = os.path.curdir
        super().__init__(iona_stream)


def iona_load_config(iona_config_path):
    with open(iona_config_path, "r") as iona_f:
        iona_config = EasyDict(yaml.load(iona_f, IONAConfigLoader))
    iona_config_name = os.path.basename(iona_config_path)[
        : os.path.basename(iona_config_path).rfind(".")
    ]
    return (iona_config, iona_config_name)


def iona_per_complex_corr(iona_df, iona_pred_attr="ddG_pred", iona_limit=10):
    iona_corr_table = []
    for iona_cplx in iona_df["complex"].unique():
        iona_df_cplx = iona_df.query(f'complex == "{iona_cplx}"')
        if len(iona_df_cplx) < iona_limit:
            continue
        iona_corr_table.append(
            {
                "complex": iona_cplx,
                "pearson": iona_df_cplx[["ddG", iona_pred_attr]]
                .corr("pearson")
                .iloc[0, 1],
                "spearman": iona_df_cplx[["ddG", iona_pred_attr]]
                .corr("spearman")
                .iloc[0, 1],
            }
        )
    iona_corr_table = pd.DataFrame(iona_corr_table)
    iona_avg = iona_corr_table[["pearson", "spearman"]].mean()
    return (iona_avg["pearson"], iona_avg["spearman"])


def iona_overall_correlations(iona_df):
    iona_pearson = iona_df[["ddG", "ddG_pred"]].corr("pearson").iloc[0, 1]
    iona_spearman = iona_df[["ddG", "ddG_pred"]].corr("spearman").iloc[0, 1]
    return {"overall_pearson": iona_pearson, "overall_spearman": iona_spearman}


def iona_percomplex_correlations(iona_df, iona_return_details=False):
    iona_corr_table = []
    for iona_cplx in iona_df["complex"].unique():
        iona_df_cplx = iona_df.query(f'complex == "{iona_cplx}"')
        if len(iona_df_cplx) < 10:
            continue
        iona_corr_table.append(
            {
                "complex": iona_cplx,
                "pearson": iona_df_cplx[["ddG", "ddG_pred"]].corr("pearson").iloc[0, 1],
                "spearman": iona_df_cplx[["ddG", "ddG_pred"]]
                .corr("spearman")
                .iloc[0, 1],
            }
        )
    iona_corr_table = pd.DataFrame(iona_corr_table)
    if len(iona_corr_table) == 0:
        iona_out = {
            "percomplex_pearson": float("nan"),
            "percomplex_spearman": float("nan"),
        }
        if iona_return_details:
            return (iona_out, iona_corr_table)
        return iona_out
    iona_average = iona_corr_table[["pearson", "spearman"]].mean()
    iona_out = {
        "percomplex_pearson": iona_average["pearson"],
        "percomplex_spearman": iona_average["spearman"],
    }
    if iona_return_details:
        return (iona_out, iona_corr_table)
    else:
        return iona_out


def iona_overall_auroc(iona_df):
    iona_y = (iona_df["ddG"] > 0).to_numpy()
    if len(np.unique(iona_y)) < 2:
        return {"auroc": float("nan")}
    iona_score = roc_auc_score(iona_y, iona_df["ddG_pred"].to_numpy())
    return {"auroc": iona_score}


def iona_overall_rmse_mae(iona_df):
    iona_true = iona_df["ddG"].to_numpy()
    iona_pred = iona_df["ddG_pred"].to_numpy()[:, None]
    iona_reg = LinearRegression().fit(iona_pred, iona_true)
    iona_pred_corrected = iona_reg.predict(iona_pred)
    iona_rmse = np.sqrt(((iona_true - iona_pred_corrected) ** 2).mean())
    iona_mae = np.abs(iona_true - iona_pred_corrected).mean()
    return {"rmse": iona_rmse, "mae": iona_mae}


def iona_analyze_all_results(iona_df):
    iona_methods = iona_df["method"].unique()
    iona_funcs = [
        iona_overall_correlations,
        iona_overall_rmse_mae,
        iona_overall_auroc,
        iona_percomplex_correlations,
    ]
    iona_analysis = []
    for iona_method in tqdm(iona_methods):
        iona_df_this = iona_df[iona_df["method"] == iona_method]
        iona_result = {"method": iona_method}
        for iona_f in iona_funcs:
            iona_result.update(iona_f(iona_df_this))
        iona_analysis.append(iona_result)
    iona_analysis = pd.DataFrame(iona_analysis)
    return iona_analysis


def iona_analyze_all_percomplex_correlations(iona_df):
    iona_methods = iona_df["method"].unique()
    iona_df_corr = []
    for iona_method in tqdm(iona_methods):
        iona_df_this = iona_df[iona_df["method"] == iona_method]
        _, iona_df_corr_this = iona_percomplex_correlations(
            iona_df_this, iona_return_details=True
        )
        iona_df_corr_this["method"] = iona_method
        iona_df_corr.append(iona_df_corr_this)
    iona_df_corr = [iona_d for iona_d in iona_df_corr if len(iona_d) > 0]
    if len(iona_df_corr) == 0:
        return pd.DataFrame(columns=["complex", "pearson", "spearman", "method"])
    iona_df_corr = pd.concat(iona_df_corr).reset_index()
    return iona_df_corr


def iona_evaluate_mutations(iona_df_items, iona_mode, iona_ddg_cutoff=None):
    assert iona_mode in ("all", "single", "multiple")
    if iona_mode == "single":
        iona_df_items = iona_df_items.query("num_muts == 1")
    elif iona_mode == "multiple":
        iona_df_items = iona_df_items.query("num_muts > 1")
    if iona_ddg_cutoff is not None:
        iona_df_items = iona_df_items.query(
            f"ddG >= {-iona_ddg_cutoff} and ddG <= {iona_ddg_cutoff}"
        )
    if len(iona_df_items) == 0:
        iona_nanrow = {
            "method": "RDE",
            "overall_pearson": float("nan"),
            "overall_spearman": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "auroc": float("nan"),
            "percomplex_pearson": float("nan"),
            "percomplex_spearman": float("nan"),
            "mode": iona_mode,
        }
        return pd.DataFrame([iona_nanrow])
    iona_df_metrics = iona_analyze_all_results(iona_df_items)
    iona_df_corr = iona_analyze_all_percomplex_correlations(iona_df_items)
    iona_df_metrics["mode"] = iona_mode
    return iona_df_metrics


def iona_evaluate_mutation_subsets(iona_results, iona_ddg_cutoff=None):
    iona_df_all = iona_evaluate_mutations(
        iona_results, iona_mode="all", iona_ddg_cutoff=iona_ddg_cutoff
    )
    iona_df_single = iona_evaluate_mutations(
        iona_results, iona_mode="single", iona_ddg_cutoff=iona_ddg_cutoff
    )
    iona_df_multiple = iona_evaluate_mutations(
        iona_results, iona_mode="multiple", iona_ddg_cutoff=iona_ddg_cutoff
    )
    iona_df_metrics = pd.concat([iona_df_all, iona_df_single, iona_df_multiple], axis=0)
    iona_df_metrics.reset_index(inplace=True, drop=True)
    return iona_df_metrics
