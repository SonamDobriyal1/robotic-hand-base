# RGB-D Robotic Grasp Prediction MVP

This is a small PyTorch research codebase for vision-based robotic grasp prediction on the Cornell Grasp Dataset. It predicts dense maps for:

- grasp quality
- grasp angle
- grasp gripper opening width

The MVP is intentionally simple: no ROS, no robot hardware, and no simulator dependency. PyBullet can be added later for synthetic data or closed-loop testing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

Download and extract the Cornell Grasp Dataset, then place or symlink it under:

```text
data/cornell/
```

The loader recursively searches for standard Cornell files:

```text
pcd0100r.png
pcd0100d.tiff
pcd0100cpos.txt
```

If your dataset is somewhere else, update `data.root` in `configs/baseline.yaml`.
Samples are shuffled deterministically with the configured seed and split into
70% training, 15% validation, and 15% test sets by default. The validation and
test fractions can be changed with `data.val_split` and `data.test_split`.

## Train

```bash
python train.py --config configs/baseline.yaml
```

Training writes:

```text
checkpoints/best.pt
checkpoints/last.pt
outputs/train_log.csv
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split test
```

Accuracy is counted as correct when the best predicted grasp rectangle passes both:

- rectangle IoU threshold, default `0.25`
- angle threshold, default `30` degrees

These settings live in `configs/baseline.yaml`.

## Visualize

```bash
python visualize.py --checkpoint checkpoints/best.pt --sample_id 0
```

This saves an overlay image to:

```text
outputs/prediction_sample_0.png
```

The red rectangle is the model prediction. Green rectangles are Cornell ground-truth positive grasps.

## Smoke Test

The smoke test uses fake tensors, so it does not require the Cornell dataset:

```bash
python tests/smoke_test.py
```

## Project Structure

```text
data/
  cornell.py          Cornell dataset parser and dense target-map generation
models/
  ggcnn.py            Lightweight GG-CNN-inspired baseline CNN
utils/
  config.py           YAML config helpers
  geometry.py         Rectangle, IoU, and angle utilities
  prediction.py       Converts dense maps into one grasp proposal
configs/
  baseline.yaml       Default experiment config
train.py              Training entry point
evaluate.py           Evaluation entry point
visualize.py          Visualization entry point
tests/smoke_test.py   Minimal fake-data smoke test
```

## Notes

The model directly regresses angle in radians and normalized width in `[0, 1]`. A stronger next version would predict `cos(2theta)` and `sin(2theta)` to avoid angle wraparound, add data augmentation, and use a larger validation protocol.
