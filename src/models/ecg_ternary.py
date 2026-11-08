from torch.nn.utils import parametrize

from src.models.ecg_fp32 import ECGFP32
from src.quantization.ternary import TernaryWeight


class ECGTernary(ECGFP32):
    """Reuse the FP32 topology and apply a ternary forward pass to every weight."""
    weight_layers = ("conv1", "conv2", "head")

    def __init__(self):
        super().__init__()
        for name in self.weight_layers:
            parametrize.register_parametrization(getattr(self, name), "weight", TernaryWeight())
