from pathlib import Path
from typing import Sequence

import cv2

from association_quality_clavia import AssociationQuality
from binary_classification_ratios import BinaryClassificationRatios
from trackers import load_mot_file
from trackers.io.mot import _mot_frame_to_detections

from eval_roboflow_trackers.bytetrack.tracker_ia import ByteTrackTrackerIa

from .clavia_annotator import ClavIaAnnotator
from .cli import get_cmd_line


def run(args: Sequence[str] | None = None) -> int:
    cli = get_cmd_line(args)
    paths = Path.cwd().rglob(cli.glob)
    aq_global = AssociationQuality()
    for path in sorted(paths):
        mot_frames = load_mot_file(path)
        tracker = ByteTrackTrackerIa()
        aq_local = AssociationQuality()

        if cli.annotate_fp:
            cv2.namedWindow('frame', cv2.WINDOW_GUI_EXPANDED)
            ann = ClavIaAnnotator()

        for frame_idx in sorted(mot_frames):
            gt = _mot_frame_to_detections(mot_frames[frame_idx])
            gt.detection_id = mot_frames[frame_idx].ids
            tracker.update(gt)

            for t in tracker.tracks:
                aq_local.classify(t.ann_id, t.upd_id, t.ann_id in gt.detection_id)
                aq_global.classify(t.ann_id, t.upd_id, t.ann_id in gt.detection_id)

            if cli.annotate_fp:
                image = cv2.imread(path.parent.parent / f'img1/{frame_idx:06d}.jpg')
                num_fp = ann.annotate(image, tracker.tracks, gt.detection_id)
                cv2.imshow('frame', image)
                cv2.waitKey(1000 if num_fp > 0 else 10)

        if cli.verbosity > 0:
            bcr_local = BinaryClassificationRatios(**aq_local.get_confusion_matrix())
            print(path, bcr_local.get_summary_dct()['f1_score'])

    bcr_global = BinaryClassificationRatios(**aq_global.get_confusion_matrix())
    print(bcr_global.get_summary())
    return 0


def main() -> None:
    run()
