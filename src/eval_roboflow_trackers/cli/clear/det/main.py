from pathlib import Path
from typing import Sequence

from binary_classification_ratios import BinaryClassificationRatios
from trackers import ByteTrackTracker, load_mot_file
from trackers.eval import compute_clear_metrics, compute_hota_metrics
from trackers.io.mot import _prepare_mot_sequence

from eval_roboflow_trackers.cli.clear.f1_scores import get_f1_clear, get_f1_hota
from eval_roboflow_trackers.cli.clear.keeper import Keeper
from eval_roboflow_trackers.cli.clear.track_write_load import track_write_load

from .cli import get_cmd_line


def run(args: Sequence[str] | None = None) -> int:
    cli = get_cmd_line(args)
    det_paths = sorted(Path.cwd().rglob(cli.glob_det))
    gt_paths = sorted(Path.cwd().rglob(cli.glob_gt))
    global_cm_clear = Keeper()
    global_cm_hota = Keeper()
    for path_idx, det_path in enumerate(det_paths):
        tracker = ByteTrackTracker()
        tracker_data = track_write_load(tracker, load_mot_file(det_path))

        gt_data = load_mot_file(gt_paths[path_idx])
        sd = _prepare_mot_sequence(
            gt_data, tracker_data
        )  # Prepare sequence (compute IoU, remap IDs)

        clear_dict = compute_clear_metrics(sd.gt_ids, sd.tracker_ids, sd.similarity_scores)
        hota_dict = compute_hota_metrics(sd.gt_ids, sd.tracker_ids, sd.similarity_scores)

        global_cm_clear.add_clear(**clear_dict)
        global_cm_hota.add_hota(**hota_dict)
        if cli.verbosity > 0:
            print(det_path, get_f1_clear(clear_dict), get_f1_hota(hota_dict))

    bcr_global_clear = BinaryClassificationRatios(**vars(global_cm_clear))
    print(bcr_global_clear.get_summary())

    bcr_global_hota = BinaryClassificationRatios(**vars(global_cm_hota))
    print(bcr_global_hota.get_summary())

    return 0


def main() -> None:
    run()
