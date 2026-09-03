from trackers.io.mot import load_mot_file


det = load_mot_file('data/mot17/val/MOT17-02-FRCNN/det/det.txt')
gt = load_mot_file('data/mot17/val/MOT17-02-FRCNN/gt/gt.txt')

breakpoint()
