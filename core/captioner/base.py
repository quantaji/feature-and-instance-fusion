import torch


class BaseCaptioner:
    def load_model(self):
        raise NotImplementedError

    def get_caption(
        self,
        feat: torch.Tensor,
        normalize: bool = True,
    ) -> str:
        raise NotImplementedError
