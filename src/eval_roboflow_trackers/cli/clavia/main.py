from pathlib import Path
from typing import Sequence

from association_quality_clavia import AssociationQuality
from binary_classification_ratios import BinaryClassificationRatios
from trackers import load_mot_file

from eval_roboflow_trackers.bytetrack.tracker_ia import ByteTrackTrackerIa

from .cli import get_cmd_line
from .track_gt import advance_tracker


def run(args: Sequence[str] | None = None) -> int:
    cli = get_cmd_line(args)
    paths = Path.cwd().rglob(cli.glob)
    aq_global = AssociationQuality()
    for path in sorted(paths):
        mot_frame_data = load_mot_file(path)
        tracker = ByteTrackTrackerIa()
        aq_local = AssociationQuality()
        for frame_idx in sorted(mot_frame_data):
            detection_id = advance_tracker(tracker, mot_frame_data[frame_idx], [1])
            for t in tracker.tracks:
                aq_local.classify(t.ann_id, t.upd_id, t.ann_id in detection_id)
                aq_global.classify(t.ann_id, t.upd_id, t.ann_id in detection_id)
        bcr_local = BinaryClassificationRatios(**aq_local.get_confusion_matrix())
        if cli.verbosity > 0:
            print(path)
            print(bcr_local.get_summary())

    bcr_global = BinaryClassificationRatios(**aq_global.get_confusion_matrix())
    print(bcr_global.get_summary())
    return 0


def main() -> None:
    run()
