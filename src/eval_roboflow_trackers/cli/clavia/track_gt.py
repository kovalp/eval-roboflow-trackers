from typing import Sequence

import numpy as np

from trackers.io.mot import _mot_frame_to_detections, _MOTFrameData

from eval_roboflow_trackers.bytetrack.tracker_ia import ByteTrackTrackerIa


def advance_tracker_selected(
    tracker: ByteTrackTrackerIa, mot_data: _MOTFrameData, classes: Sequence[int]
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """."""
    all_gt = _mot_frame_to_detections(mot_data)
    all_gt.tracker_id = mot_data.ids
    selected_gt = all_gt[np.isin(all_gt.class_id, classes)]
    selected_gt.detection_id = selected_gt.tracker_id.copy()
    selected_gt.tracker_id = None
    tracker.update(selected_gt)
    return selected_gt.detection_id


def advance_tracker_all(tracker: ByteTrackTrackerIa, mot_data: _MOTFrameData) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """."""
    all_gt = _mot_frame_to_detections(mot_data)
    all_gt.detection_id = mot_data.ids
    tracker.update(all_gt)
    return all_gt.detection_id
