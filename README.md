# Robotic Arm Simulation

A planar 3-joint robot arm (shoulder, elbow, wrist + gripper) with a pick-and-place
program, pointer-following inverse kinematics and manual jogging, plus two trained
networks: a graph network that solves the arm's inverse kinematics, and a vision
network that finds a part in a camera image.

In **Vision** mode the two work together. A camera images the floor, the vision
network finds the part, the GNN turns each gripper target into joint angles, and the
arm drops the part in a bin. If the gripper closes on nothing, the arm backs off and
looks again. Click the floor to move the part.

Serve the folder and open it in a browser (see below). `ik-gnn-weights.js` and
`vision-weights.js` must sit next to `index.html` for the **GNN** solver and the
**Vision** mode to work.

## The GNN

`gnn/` trains a graph network that maps a gripper target (x, z, tool angle,
elbow up/down) to the three joint angles. The graph is the arm itself:

```
shoulder ── elbow ── wrist ── tcp        chain edges carry the link length
    └─────────┴────────┴───────┘
              goal                       goal node carries the target
```

Each joint node decodes its own angle after 5 rounds of message passing.
Training data is generated on the fly by sampling joint angles inside the limits
and running forward kinematics, so every sample is exact and the set is unlimited.
The loss combines joint-angle error with gripper position error through a
differentiable forward-kinematics layer. The forward-kinematics terms ramp in over
the first 30% of training: switched on at full weight from the start, their huge
early errors push the outputs into wrong poses that training never escapes. Each
joint's output goes through `tanh`, so it always stays inside that joint's limits.

The elbow flag is an input because every target has two valid solutions
(elbow up and elbow down); without it the mapping would not be a function.

With the **GNN** solver selected, the network commands the joints at every control
step (240 per second) of every move. Straight-line moves (LIN) ask it for each point
along the line. Joint moves (PTP) are still planned by blending the joints to the end
pose, but each step's commanded angles are the GNN's prediction for where the gripper
should be at that moment; the dashed outline shows the planned pose. Near a straight
arm the network keeps the elbow a few degrees off straight on each branch, so where a
joint move flips the elbow over, the page blends the elbow-up and elbow-down
predictions across a 6° band instead of jumping ~7° in one step.

## The vision network

`vision/` trains a small fully convolutional network (26k weights) that finds the
part in a 160×40 camera image of the floor strip x 100–660 mm (3.5 mm per pixel). It
outputs a score map, and a soft-argmax turns that into the part's (x, z) in mm.

Training scenes are rendered on the GPU on the fly: one part in a random spot and
colour, the arm partly in view in some scenes, and random lighting, colour balance
and sensor noise. The page renders its camera image from the same constants, which
`vision/export.py` ships with the weights, so the network sees in the browser what
it was trained on. The loss is Huber rather than squared error: when the arm hides
the part, the miss can be hundreds of mm, and a squared loss lets that one scene
derail training.

## Setup

The venv lives on the home drive because this USB drive is exFAT, which can't
hold the symlinks a venv needs.

```bash
python3 -m venv ~/.venvs/robotic-arm-simulation
~/.venvs/robotic-arm-simulation/bin/pip install -r requirements.txt
```

## Train, evaluate, export

```bash
cd gnn
~/.venvs/robotic-arm-simulation/bin/python train.py      # writes ik_gnn.pt
~/.venvs/robotic-arm-simulation/bin/python evaluate.py   # writes eval.json
~/.venvs/robotic-arm-simulation/bin/python export.py     # writes ../ik-gnn-weights.js
~/.venvs/robotic-arm-simulation/bin/python plot.py       # writes training_curves.png
```

`train.py` accepts `--steps`, `--batch`, `--hidden`, `--mp-steps`, `--lr` and `--device`.

The vision network follows the same pattern (about 2 minutes on an RTX 4090):

```bash
cd vision
~/.venvs/robotic-arm-simulation/bin/python train.py --device cuda   # writes vision_net.pt
~/.venvs/robotic-arm-simulation/bin/python evaluate.py              # writes eval.json
~/.venvs/robotic-arm-simulation/bin/python export.py                # writes ../vision-weights.js
```

Training runs on the CPU by default. The first CUDA calculation on this laptop's
GTX 1050 (hybrid Intel + NVIDIA graphics) coincided with a full system freeze, so
the GPU stays off until that is understood. `--device cuda` turns it back on. On 4
CPU threads the default model takes about 0.22 s per step at batch 1024, so
`python train.py --steps 12000 --batch 1024` takes roughly 45 minutes. On a
desktop RTX 4090, the full default run (`--device cuda`, 30k steps, batch 4096)
takes about 7 minutes.

The page needs a web server to load the weight files; opened straight from
disk, some browsers block the scripts. From the project folder:
`python -m http.server 8765`, then open http://localhost:8765.

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The simulation, including JavaScript ports of both networks and the camera |
| `ik-gnn-weights.js` | Exported GNN weights (generated) |
| `vision-weights.js` | Exported vision weights and camera definition (generated) |
| `gnn/arm.py` | Arm geometry, forward kinematics, closed-form IK, data sampling |
| `gnn/model.py` | The graph network (plain PyTorch, no PyG) |
| `gnn/train.py` | GPU training loop |
| `gnn/evaluate.py` | Accuracy against closed-form IK on random and program targets |
| `gnn/export.py` | Weight export for the web page |
| `gnn/plot.py` | Training curves from `train_history.json` (written by `train.py`) |
| `vision/scene.py` | The camera: view, scene renderer, random training scenes |
| `vision/model.py` | The vision network (convolutions + soft-argmax) |
| `vision/train.py` | Training loop |
| `vision/evaluate.py` | Position error on unseen scenes, and share close enough to grip |
| `vision/export.py` | Weight and camera export for the web page |
| `CHAT.md` | Cleaned log of the Claude Code session that built this project |

## Results

The default 30k-step run, evaluated on 200,000 unseen targets: gripper error
0.29 mm mean, 0.65 mm p95, 4.4 mm max (100% within 5 mm); tool angle error
0.06° mean. Every pick-and-place waypoint is within 0.25 mm. Inference in the
browser takes about 3 ms.

The vision network, on 10,000 unseen scenes: position error 0.28 mm mean, 0.70 mm
p95, 10.4 mm max; every part found close enough to grip (the gripper catches a part
within 22 mm). With lighting and noise variation added it stays at 0.39 mm mean, with
3 in 10,000 out of reach of the gripper, all pale parts on a background lit to almost
the same colour. In the browser it finds the part to within about 0.1–0.4 mm in
about 50 ms, and in a test run Vision mode binned 30 parts with no misses other than
one caused by moving the part while the arm was reaching for it.
