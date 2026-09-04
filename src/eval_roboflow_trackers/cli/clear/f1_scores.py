from binary_classification_ratios import BinaryClassificationRatios

from eval_roboflow_trackers.cli.clear.keeper import Keeper


def get_f1_clear(dct: dict[str, float]) -> float:
    cm = Keeper()
    cm.add_clear(**dct)
    return BinaryClassificationRatios(**vars(cm)).get_summary_dct()['f1_score']


def get_f1_hota(dct: dict[str, float]) -> float:
    cm = Keeper()
    cm.add_hota(**dct)
    return BinaryClassificationRatios(**vars(cm)).get_summary_dct()['f1_score']
