from supervision import BoxAnnotator, LabelAnnotator


class MyAnnotator(object):
    def __init__(self):
        self.label_ann = LabelAnnotator()
        self.box_ann = BoxAnnotator()

    def annotate(self, image, detections) -> None:
        self.label_ann.annotate(image, detections, labels=detections.tracker_id)
        self.box_ann.annotate(image, detections)
