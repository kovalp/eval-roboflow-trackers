import cv2
import numpy as np

from supervision.annotators.core import BoxAnnotator, LabelAnnotator
from trackers import load_mot_file
from trackers.io.mot import _mot_frame_to_detections


mot_frame_data = load_mot_file('data/mot17/val/MOT17-02-FRCNN/gt/gt.txt')

label_ann = LabelAnnotator()
box_ann = BoxAnnotator()

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, 1)


for frame_idx in sorted(mot_frame_data):
    image = cv2.imread(f'data/mot17/val/MOT17-02-FRCNN/img1/{frame_idx:06d}.jpg')
    mot_data = mot_frame_data[frame_idx]
    all_targets = _mot_frame_to_detections(mot_data)
    all_targets.tracker_id = mot_data.ids
    mask = np.isin(all_targets.class_id, [1])
    pedestrians = all_targets[mask]

    label_ann.annotate(image, pedestrians, labels=pedestrians.tracker_id)
    box_ann.annotate(image, pedestrians)
    cv2.imshow('frame', image)
    cv2.waitKey(10)
