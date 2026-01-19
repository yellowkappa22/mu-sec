from torch import nn

class LinearPredicterModel(nn.Module):
    super()
class LinearPredicterStrategy:
    def __init__(self, window_size: int, entry_thr: float, exit_thr: float):
        self.window_size = window_size
        self.entry_thr = entry_thr
        self.exit_thr = exit_thr