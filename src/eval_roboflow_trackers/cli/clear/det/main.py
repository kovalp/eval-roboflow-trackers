from pathlib import Path
from typing import Sequence

from binary_classification_ratios import BinaryClassificationRatios
from trackers import ByteTrackTracker, load_mot_file
from trackers.eval import CLEARMetrics, compute_clear_metrics
from trackers.io.mot import _mot_frame_to_detections, _MOTOutput, _prepare_mot_sequence

from .cli import get_cmd_line


def run(args: Sequence[str] | None = None) -> int:
    cli = get_cmd_line(args)
    threshold = 0.5
    det_paths = sorted(Path.cwd().rglob(cli.glob_det))
    gt_paths = sorted(Path.cwd().rglob(cli.glob_gt))
    global_tp, global_fn, global_fp = 0, 0, 0
    for path_idx, det_path in enumerate(det_paths):
        mot_frame_data = load_mot_file(det_path)
        tracker = ByteTrackTracker()
        with _MOTOutput(Path('tmp-dir-track-eval/tracks.txt')) as mot:
            for frame_idx in sorted(mot_frame_data):
                detections_data = mot_frame_data[frame_idx]
                detections = _mot_frame_to_detections(detections_data)
                tracked = tracker.update(detections)
                mot.write(frame_idx, tracked)

        gt_data = load_mot_file(gt_paths[path_idx])
        tracker_data = load_mot_file('tmp-dir-track-eval/tracks.txt')

        # Prepare sequence (compute IoU, remap IDs)
        seq_data = _prepare_mot_sequence(gt_data, tracker_data)
        clear_metrics_dict = compute_clear_metrics(
            seq_data.gt_ids, seq_data.tracker_ids, seq_data.similarity_scores, threshold=threshold
        )
        clear = CLEARMetrics.from_dict(clear_metrics_dict)
        global_tp += clear.CLR_TP
        global_fn += clear.CLR_FN
        global_fp += clear.CLR_FP
        if cli.verbosity > 0:
            print(det_path)
            bcr_local = BinaryClassificationRatios(
                tp=clear.CLR_TP, fn=clear.CLR_FN, fp=clear.CLR_FP, tn=0
            )
            print(bcr_local.get_summary())

    bcr_global = BinaryClassificationRatios(tp=global_tp, fn=global_fn, fp=global_fp, tn=0)
    print(bcr_global.get_summary())

    return 0


def main() -> None:
    run()
