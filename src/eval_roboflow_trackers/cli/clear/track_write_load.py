from pathlib import Path

from trackers import ByteTrackTracker
from trackers.io.mot import _mot_frame_to_detections, _MOTFrameData, _MOTOutput, load_mot_file


def track_write_load(
    tracker: ByteTrackTracker, mot_frames: dict[int, _MOTFrameData]
) -> dict[int, _MOTFrameData]:
    with _MOTOutput(Path('tmp-dir-track-eval/tracks.txt')) as mot:
        for frame_idx in sorted(mot_frames):
            detections = _mot_frame_to_detections(mot_frames[frame_idx])
            detections.tracker_id = None
            tracked = tracker.update(detections)
            mot.write(frame_idx, tracked)
    return load_mot_file('tmp-dir-track-eval/tracks.txt')
