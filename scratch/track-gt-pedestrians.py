import cv2
import numpy as np

from trackers import load_mot_file
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.io.mot import _mot_frame_to_detections

from eval_roboflow_trackers.my_annotator import MyAnnotator


mot_frame_data = load_mot_file('data/mot17/val/MOT17-02-FRCNN/gt/gt.txt')

ann = MyAnnotator()

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
## cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
tracker = ByteTrackTracker()


for frame_idx in sorted(mot_frame_data):
    image = cv2.imread(f'data/mot17/val/MOT17-02-FRCNN/img1/{frame_idx:06d}.jpg')
    mot_data = mot_frame_data[frame_idx]
    all_gt = _mot_frame_to_detections(mot_data)
    all_gt.tracker_id = mot_data.ids
    pedestrians_gt = all_gt[np.isin(all_gt.class_id, [1])]
    pedestrians_gt.tracker_id = None
    targets = tracker.update(pedestrians_gt)

    ann.annotate(image, targets)
    print(targets.tracker_id)
    cv2.imshow('frame', image)
    if cv2.waitKey(10) == ord('q'):
        break
