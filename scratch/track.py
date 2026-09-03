import cv2

from supervision.annotators.core import BoxAnnotator, LabelAnnotator, TraceAnnotator
from trackers import load_mot_file
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.io.mot import _mot_frame_to_detections


mot_frame_data = load_mot_file('data/mot17/val/MOT17-02-FRCNN/det/det.txt')
tracker = ByteTrackTracker()

trace_ann = TraceAnnotator()
label_ann = LabelAnnotator()
box_ann = BoxAnnotator()

for frame_idx in sorted(mot_frame_data):
    image = cv2.imread(f'data/mot17/val/MOT17-02-FRCNN/img1/{frame_idx:06d}.jpg')
    detections_data = mot_frame_data[frame_idx]
    detections = _mot_frame_to_detections(detections_data)
    targets = tracker.update(detections)
    label_ann.annotate(image, targets, labels=targets.tracker_id)
    box_ann.annotate(image, targets)
    cv2.imshow('frame', image)
    cv2.waitKey(20)
