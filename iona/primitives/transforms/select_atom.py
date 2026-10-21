from ._base import iona_register_transform


@iona_register_transform("select_atom")
class IONASelectAtom(object):

    def __init__(self, iona_resolution):
        super().__init__()
        assert iona_resolution in ("full", "backbone", "backbone+CB")
        self.iona_resolution = iona_resolution

    def __call__(self, iona_data):
        if self.iona_resolution == "full":
            iona_data["pos_atoms"] = iona_data["pos_heavyatom"][:, :]
            iona_data["mask_atoms"] = iona_data["mask_heavyatom"][:, :]
        elif self.iona_resolution == "backbone":
            iona_data["pos_atoms"] = iona_data["pos_heavyatom"][:, :4]
            iona_data["mask_atoms"] = iona_data["mask_heavyatom"][:, :4]
        elif self.iona_resolution == "backbone+CB":
            iona_data["pos_atoms"] = iona_data["pos_heavyatom"][:, :5]
            iona_data["mask_atoms"] = iona_data["mask_heavyatom"][:, :5]
        return iona_data
