from pathlib import Path
from typing import Sequence

from binary_classification_ratios import BinaryClassificationRatios
from trackers import ByteTrackTracker, load_mot_file
from trackers.eval import compute_clear_metrics, compute_hota_metrics
from trackers.io.mot import _prepare_mot_sequence

from ..f1_scores import get_f1_clear, get_f1_hota
from ..keeper import Keeper
from ..track_write_load import track_write_load
from .cli import get_cmd_line


def run(args: Sequence[str] | None = None) -> int:
    cli = get_cmd_line(args)
    gt_paths = sorted(Path.cwd().rglob(cli.glob))
    global_cm_clear = Keeper()
    global_cm_hota = Keeper()
    for gt_path in gt_paths:
        mot_frame_data = load_mot_file(gt_path)
        tracker = ByteTrackTracker()
        tracker_data = track_write_load(tracker, mot_frame_data)
        gt_data = load_mot_file(gt_path)
        sd = _prepare_mot_sequence(gt_data, tracker_data)
        clear_dict = compute_clear_metrics(sd.gt_ids, sd.tracker_ids, sd.similarity_scores)
        hota_dict = compute_hota_metrics(sd.gt_ids, sd.tracker_ids, sd.similarity_scores)

        global_cm_clear.add_clear(**clear_dict)
        global_cm_hota.add_hota(**hota_dict)
        if cli.verbosity > 0:
            print(gt_path, get_f1_clear(clear_dict), get_f1_hota(hota_dict))

    print(BinaryClassificationRatios(**vars(global_cm_clear)).get_summary())
    print(BinaryClassificationRatios(**vars(global_cm_hota)).get_summary())
    return 0


def main() -> None:
    run()
