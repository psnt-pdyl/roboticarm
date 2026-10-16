# Robotic Arm Simulation

A planar 3-joint robot arm (shoulder, elbow, wrist + gripper) with a pick-and-place
program, pointer-following inverse kinematics and manual jogging, plus a graph
neural network trained to solve the arm's inverse kinematics.

Open `index.html` in a browser. `ik-gnn-weights.js` must sit next to it for the
**GNN** solver button to work.

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
differentiable forward-kinematics layer.

The elbow flag is an input because every target has two valid solutions
(elbow up and elbow down); without it the mapping would not be a function.

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
```

`train.py` accepts `--steps`, `--batch`, `--hidden`, `--mp-steps`, `--lr` and `--device`.

Training runs on the CPU by default. The first CUDA calculation on this laptop's
GTX 1050 (hybrid Intel + NVIDIA graphics) coincided with a full system freeze, so
the GPU stays off until that is understood. `--device cuda` turns it back on. On 4
CPU threads the default model takes about 0.22 s per step at batch 1024, so
`python train.py --steps 12000 --batch 1024` takes roughly 45 minutes.

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The simulation, including a JavaScript port of the GNN forward pass |
| `ik-gnn-weights.js` | Exported weights (generated) |
| `gnn/arm.py` | Arm geometry, forward kinematics, closed-form IK, data sampling |
| `gnn/model.py` | The graph network (plain PyTorch, no PyG) |
| `gnn/train.py` | GPU training loop |
| `gnn/evaluate.py` | Accuracy against closed-form IK on random and program targets |
| `gnn/export.py` | Weight export for the web page |
| `CHAT.md` | Cleaned log of the Claude Code session that built this project |

The GNN has not been trained yet, so `ik-gnn-weights.js` is not in the repo.
