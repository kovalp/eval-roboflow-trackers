import cv2
import numpy as np

from association_quality_clavia import AssociationQuality
from binary_classification_ratios import BinaryClassificationRatios
from trackers import load_mot_file
from trackers.io.mot import _mot_frame_to_detections

from eval_roboflow_trackers.bytetrack.tracker_ia import ByteTrackTrackerIa
from eval_roboflow_trackers.my_annotator import MyAnnotator


root_dir = 'data/mot17/val/MOT17-02-FRCNN'
mot_frame_data = load_mot_file(f'{root_dir}/gt/gt.txt')

ann = MyAnnotator()

## cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
## cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
tracker = ByteTrackTrackerIa()
aq = AssociationQuality()

for frame_idx in sorted(mot_frame_data):
    image = cv2.imread(f'{root_dir}/img1/{frame_idx:06d}.jpg')
    mot_data = mot_frame_data[frame_idx]
    all_gt = _mot_frame_to_detections(mot_data)
    all_gt.tracker_id = mot_data.ids
    pedestrians_gt = all_gt[np.isin(all_gt.class_id, [1])]
    pedestrians_gt.detection_id = pedestrians_gt.tracker_id.copy()
    pedestrians_gt.tracker_id = None
    tracker.update(pedestrians_gt)

    for t in tracker.tracks:
        aq.classify(t.ann_id, t.upd_id, t.ann_id in pedestrians_gt.detection_id)

bcr = BinaryClassificationRatios(**aq.get_confusion_matrix())
print(bcr.get_summary())

# ann.annotate(image, targets)
# print(targets.tracker_id)
# cv2.imshow("frame", image)
# if cv2.waitKey(10) == ord('q'):
#     break
