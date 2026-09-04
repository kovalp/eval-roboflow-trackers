# Evaluate using `trackers` by Roboflow

Instrumented ByteTrack tracker and scripts to evaluate the tracker with ClavIA, CLEAR and HOTA metrics.

## Installation

```shell
uv sync
```

After installation, the scripts `eval-bytetrack-clavia`, 
`eval-bytetrack-clear-detections` and `eval-bytetrack-clear-gt` become available in the shell.

## Download the Dataset

Download can be accomplished via `trackers` CLI

```shell
trackers download mot17 --split val --output ./data
```

## Evaluate via Roboflow scripts

[Download the val subset of MOT17 dataset](#download-the-dataset).

```shell
trackers track --detections ./data/mot17/val/MOT17-02-FRCNN/det/det.txt --mot-output results/MOT17-02-FRCNN.txt --overwrite
```

```shell
trackers track --detections ./data/mot17/val/MOT17-02-FRCNN/det/det.txt --mot-output results/MOT17-02-FRCNN.txt --overwrite
trackers eval --gt ./data/mot17/val/MOT17-02-FRCNN/gt/gt.txt --tracker results/MOT17-02-FRCNN.txt --metrics CLEAR
```


## Evaluate via custom scripts

[Download the val subset of MOT17 dataset](#download-the-dataset).

### ClavIA

```shell
eval-bytetrack-clavia
```

### CLEAR and HOTA

If you feed detections to the tracker

```shell
eval-bytetrack-clear-detections
```

If you feed ground truth (annotations) to the tracker

```shell
eval-bytetrack-clear-gt
```

