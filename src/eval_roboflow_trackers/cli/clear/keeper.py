class Keeper:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add_clear(self, CLR_TP=0, CLR_FN=0, CLR_FP=0, **kwargs) -> None:
        self.tp += CLR_TP
        self.fn += CLR_FN
        self.fp += CLR_FP

    def add_hota(self, HOTA_TP=0, HOTA_FN=0, HOTA_FP=0, **kwargs) -> None:
        self.tp += HOTA_TP
        self.fn += HOTA_FN
        self.fp += HOTA_FP
