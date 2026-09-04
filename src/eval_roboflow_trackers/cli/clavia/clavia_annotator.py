from typing import Sequence

import numpy as np
from association_quality_clavia import AssociationQuality
from supervision import LabelAnnotator, BoxAnnotator, Detections, Position

from eval_roboflow_trackers.bytetrack.tracklet_ia import ByteTrackTrackletIa


class ClavIaAnnotator:
    def __init__(self) -> None:
        self.show_tp = True
        self.label_ann_ann = LabelAnnotator(text_position=Position.BOTTOM_CENTER)
        self.label_ann_upd = LabelAnnotator(text_position=Position.TOP_RIGHT)
        self.box_ann = BoxAnnotator()

    def annotate(self,
                 image: np.ndarray,
                 tracks: list[ByteTrackTrackletIa],
                 detection_id: Sequence[int]
                 ) -> int:
        aq = AssociationQuality()

        boxes = []
        ann_id = []
        upd_id = []
        class_id = []
        for t in tracks:
            bin_cls = aq.classify(t.ann_id, t.upd_id, t.ann_id in detection_id)
            if bin_cls == bin_cls.FP:
                boxes.append(t.get_state_bbox())
                ann_id.append(t.ann_id)
                upd_id.append(t.upd_id)
                class_id.append(1)

        if len(boxes) > 0:
            detections = Detections(xyxy=np.array(boxes), class_id=np.array(class_id))
            detections.tracker_id = np.array(ann_id)
            self.box_ann.annotate(image, detections)
            self.label_ann_ann.annotate(image, detections, labels=ann_id)
            self.label_ann_upd.annotate(image, detections, labels=upd_id)

        return len(boxes)

