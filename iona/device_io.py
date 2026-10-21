import torch


def iona_move_to_device(iona_obj, iona_device):
    if isinstance(iona_obj, torch.Tensor):
        try:
            return iona_obj.cuda(device=iona_device, non_blocking=True)
        except RuntimeError:
            return iona_obj.to(iona_device)
    elif isinstance(iona_obj, list):
        return [
            iona_move_to_device(iona_o, iona_device=iona_device) for iona_o in iona_obj
        ]
    elif isinstance(iona_obj, tuple):
        return tuple(
            (
                iona_move_to_device(iona_o, iona_device=iona_device)
                for iona_o in iona_obj
            )
        )
    elif isinstance(iona_obj, dict):
        return {
            iona_k: iona_move_to_device(iona_v, iona_device=iona_device)
            for iona_k, iona_v in iona_obj.items()
        }
    else:
        return iona_obj


class IONACrossValidation(object):

    def __init__(self, iona_config, iona_num_cvfolds, iona_model_factory):
        super().__init__()
        self.iona_num_cvfolds = iona_num_cvfolds
        self.iona_models = [
            iona_model_factory(iona_config) for _ in range(iona_num_cvfolds)
        ]
        self.iona_optimizers = []
        self.iona_schedulers = []
        for iona_model in self.iona_models:
            iona_optimizer = torch.optim.Adam(
                iona_model.parameters(),
                lr=iona_config.iona_lr,
                weight_decay=iona_config.iona_weight_decay,
                betas=(0.9, 0.999),
            )
            iona_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                iona_optimizer, factor=0.8, patience=10, min_lr=1e-06
            )
            self.iona_optimizers.append(iona_optimizer)
            self.iona_schedulers.append(iona_scheduler)

    def iona_get(self, iona_fold):
        return (
            self.iona_models[iona_fold],
            self.iona_optimizers[iona_fold],
            self.iona_schedulers[iona_fold],
        )

    def to(self, iona_device):
        for iona_m in self.iona_models:
            iona_m.to(iona_device)
        return self

    def state_dict(self):
        return {
            "models": [iona_m.state_dict() for iona_m in self.iona_models],
            "optimizers": [iona_o.state_dict() for iona_o in self.iona_optimizers],
            "schedulers": [iona_s.state_dict() for iona_s in self.iona_schedulers],
        }

    def load_state_dict(self, state_dict):
        for iona_sd, iona_obj in zip(state_dict["models"], self.iona_models):
            iona_obj.load_state_dict(iona_sd)
        for iona_sd, iona_obj in zip(state_dict["optimizers"], self.iona_optimizers):
            iona_obj.load_state_dict(iona_sd)
        for iona_sd, iona_obj in zip(state_dict["schedulers"], self.iona_schedulers):
            iona_obj.load_state_dict(iona_sd)
