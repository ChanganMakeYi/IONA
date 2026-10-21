import argparse
import ast
import io
from pathlib import Path
import tokenize
import unittest

import torch

from iona.experiments.run_benchmarks import iona_NETS
from iona.checkpoint_io import IONA_WEIGHT_KEYS


def iona_example_config():
    return argparse.Namespace(
        iona_node_feat_dim=32,
        iona_pair_feat_dim=16,
        iona_num_layers=2,
        iona_knn=15,
        iona_cutoff=15,
        iona_num_embeddings=16,
        iona_commitment_cost=0.25,
        iona_mask_ratio=0.1,
        iona_patch_size=24,
    )


def iona_example_batch():
    torch.manual_seed(100)
    iona_batch = {
        "aa": torch.randint(0, 20, (2, 24)),
        "chain_nb": torch.tensor([[0] * 12 + [1] * 12] * 2),
        "res_nb": torch.tensor([list(range(1, 13)) * 2] * 2),
        "pos_atoms": torch.randn(2, 24, 5, 3) * 5,
        "mask_atoms": torch.ones(2, 24, 5, dtype=torch.bool),
        "chi": torch.randn(2, 24, 4),
        "chi_mask": torch.ones(2, 24, 4, dtype=torch.bool),
        "ddG": torch.tensor([1.0, -1.0]),
    }
    for iona_key in ["phi", "psi", "omega"]:
        iona_batch[iona_key] = torch.randn(2, 24)
        iona_batch[iona_key + "_mask"] = torch.ones(2, 24, dtype=torch.bool)
    iona_batch["aa_mut"] = iona_batch["aa"].clone()
    iona_batch["aa_mut"][:, 0] = (iona_batch["aa_mut"][:, 0] + 1) % 20
    iona_batch["mut_flag"] = iona_batch["aa"] != iona_batch["aa_mut"]
    return iona_batch


class IONANamespaceTests(unittest.TestCase):
    def test_iona_reference_predictions(self):
        torch.set_num_threads(1)
        iona_reference = {
            "iona_geometry": [-0.08255103975534439, -0.08518585562705994],
            "iona": [-0.08416959643363953, -0.08484993129968643],
            "iona_distance_bias": [-0.08326483517885208, -0.08537580072879791],
            "iona_gated": [-0.08419591188430786, -0.08484096825122833],
            "iona_solvation": [-0.08417624980211258, -0.08647991716861725],
            "iona_learned_radial": [-0.08474941551685333, -0.08769413828849792],
            "iona_distance_only": [-0.08476970344781876, -0.08768309652805328],
            "iona_permuted_charge": [-0.08417767286300659, -0.08651421964168549],
            "iona_unsigned_charge": [-0.08417711406946182, -0.08658156543970108],
            "iona_unscreened": [-0.08417727053165436, -0.08656765520572662],
        }
        iona_batch = iona_example_batch()
        for iona_arch, iona_expected in iona_reference.items():
            with self.subTest(iona_arch=iona_arch):
                torch.manual_seed(123)
                iona_model = iona_NETS[iona_arch](iona_example_config()).eval()
                with torch.no_grad():
                    _, iona_output = iona_model(iona_batch, None)
                torch.testing.assert_close(
                    iona_output["ddG_pred"],
                    torch.tensor(iona_expected),
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_iona_existing_checkpoint_loading(self):
        iona_batch = iona_example_batch()
        torch.manual_seed(123)
        iona_source = iona_NETS["iona"](iona_example_config()).eval()
        iona_inverse = {
            iona_new: iona_old for iona_old, iona_new in IONA_WEIGHT_KEYS.items()
        }
        iona_weights = {
            ".".join(
                iona_inverse.get(iona_part, iona_part)
                for iona_part in iona_key.split(".")
            ): iona_value
            for iona_key, iona_value in iona_source.state_dict().items()
        }
        torch.manual_seed(456)
        iona_target = iona_NETS["iona"](iona_example_config()).eval()
        iona_target.load_state_dict(iona_weights, strict=True)
        with torch.no_grad():
            _, iona_expected = iona_source(iona_batch, None)
            _, iona_actual = iona_target(iona_batch, None)
        torch.testing.assert_close(
            iona_actual["ddG_pred"], iona_expected["ddG_pred"], rtol=0, atol=0
        )
        iona_new_target = iona_NETS["iona"](iona_example_config())
        iona_new_target.load_state_dict(iona_target.state_dict(), strict=True)

    def test_iona_source_has_no_annotations(self):
        iona_root = Path(__file__).resolve().parents[1]
        for iona_folder in ["iona", "tests"]:
            for iona_path in (iona_root / iona_folder).rglob("*.py"):
                with self.subTest(iona_path=str(iona_path)):
                    iona_text = iona_path.read_text()
                    self.assertFalse(
                        any(
                            iona_token.type == tokenize.COMMENT
                            for iona_token in tokenize.generate_tokens(
                                io.StringIO(iona_text).readline
                            )
                        )
                    )
                    for iona_node in ast.walk(ast.parse(iona_text)):
                        if isinstance(
                            iona_node,
                            (
                                ast.Module,
                                ast.ClassDef,
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                            ),
                        ):
                            self.assertIsNone(ast.get_docstring(iona_node))


if __name__ == "__main__":
    unittest.main()
