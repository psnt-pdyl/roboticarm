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

## The motion PINN

`pinn/` trains a physics-informed network that shapes the arm's joint moves. The page's
arm has no masses of its own, so `pinn/dynamics.py` gives it a body (links of 6, 4 and
1.5 kg, motor limits of 80, 30 and 6 N·m, a 10 kg part) and computes motor torques
with Newton–Euler equations, checked against a Lagrangian derivation.

The network takes a start pose, an end pose, the move's duration, the payload and a
time t, and returns the joint angles at t. Its form makes the start and end exact (right
poses, zero speed and acceleration), so it only shapes the path between. It is trained
with no example motions: velocities and accelerations come from autograd through t,
the equations of motion turn them into torques, and the loss is motor effort
(Σ (τ / τ_max)²) plus stiff penalties for exceeding torque limits, coming within 1° of a
joint limit, or bringing the elbow, wrist or tool within 20 mm of the floor.

In the page, **Joint-move motion: PINN** makes every joint move follow the network for
the same duration the joint blend would take; with the GNN solver on, the GNN still
commands each step from the PINN's pose. Moves shorter than 0.15 s stay a blend: the
network's small path correction, squeezed into so little time, would need huge torques.
The telemetry shows motor torque as a percentage of each limit, live and as the peak of
the last joint move.

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

The motion PINN (about 15 minutes on an RTX 4090):

```bash
cd pinn
python train.py --device cuda   # writes pinn.pt, train_history.json
python evaluate.py              # writes eval.json, example_move.png (compares blend, PINN, per-move optimum)
python export.py                # writes ../pinn-weights.js

# the two kept models, as used by compare.html
python export.py --model model_12kg --payload-max 12
python export.py --model model_1kg --payload-max 1 --out ../pinn-1kg-weights.js --name MOTION_PINN_1KG
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
| `compare.html` | Three synchronised cells: first PINN with the 0.4 kg part, first PINN with 10 kg, retrained PINN with 10 kg |
| `pinn-weights.js` | Exported retrained PINN (0–12 kg) and the arm's physics constants (generated) |
| `pinn-1kg-weights.js` | Exported first PINN (0–1 kg), for `compare.html` (generated) |
| `pinn/model_1kg/`, `pinn/model_12kg/` | Each kept model's results: `eval.json`, `train_history.json`, `example_move.png` (and its checkpoint, not in git) |
| `pinn/dynamics.py` | Masses, torque limits, Newton–Euler inverse dynamics, the page's joint blend |
| `pinn/model.py` | The motion PINN: q(t) with exact start and end, derivatives by autograd |
| `pinn/train.py` | Physics-only training loop and loss |
| `pinn/evaluate.py` | Blend vs PINN vs per-move optimum, on random and page moves |
| `pinn/export.py` | Weight and physics export for the web page |
| `CHAT.md` | Cleaned log of the Claude Code session that built this project |

## Results

The default 30k-step run, evaluated on 200,000 unseen targets: gripper error
0.29 mm mean, 0.65 mm p95, 4.4 mm max (100% within 5 mm); tool angle error
0.06° mean. Every pick-and-place waypoint is within 0.25 mm. Inference in the
browser takes about 3 ms.

The motion PINN, on 1,000 unseen moves (50–200 % speed, 0–1 kg payload), against the
joint blend over the same durations: 25 % less motor effort, no move past a joint limit
or into the floor (the blend drives the arm into the floor on 3 % of these moves, up to
315 mm deep), and torque limits exceeded on 0.2 % of moves (blend 0.4 %). Peak torque is
higher, 65 % of the limit on average against 54 %: the network trades a sharper moment
for an easy middle of the move, where it typically holds the arm upright, balanced over
the shoulder. A per-move optimisation of the same loss reaches 39 % less effort, so the
single network is 21 % above the optimum, and it answers 1,000 moves in 0.3 s where the
optimisation takes about 90 s. On the page's own moves at 100 % speed it saves 18–43 %;
on very short, fast moves (0.4 s) it uses more effort than the blend, because it starts
and stops without a torque jump.

The part weighs 10 kg, more than these motors can carry: just holding it still above
the bin needs 127 % of the elbow's limit, and at the far pick spot 118 % of the
shoulder's and 114 % of the elbow's. The sim does not enforce motor limits, so the arm
still moves; the torque readouts and chart show the overload in red.

The PINN is trained on payloads from 0 to 12 kg. No motion can keep 10 kg within these
motors' limits, so it learns the motion that overloads them least: carrying the part
to the bin it peaks at 157–180 % of the elbow's limit, against 255–285 % for the blend,
with about 55 % less effort. On 1,000 random moves with 0–12 kg it uses 45 % less effort
than the blend. Two costs come with the heavy range. On light moves it is weaker than
the earlier model trained on 0–1 kg (kept in `pinn/backup_1kg/`): empty moves save only
2 % effort instead of 40 %, and at 200 % speed it can use more than the blend. And where
a heavy payload makes overload unavoidable, the loss trades it against the joint limits:
10.8 % of random moves pass a joint limit, by up to 11° (the per-move optimum does the
same on 9.5 %, up to 10°). The page's own moves stay inside the joint limits and at least
42 mm above the floor.

Both models are kept (`pinn/model_1kg/`, `pinn/model_12kg/`), and `compare.html` runs
them side by side on the same pick-and-bin cycle, in step: the first PINN with the
original 0.4 kg part, the first PINN with the 10 kg part, and the retrained PINN with the
10 kg part, each with its own torque trace, the carry move's peak torque and effort,
joint-limit and floor checks. With the part at X 450 at 100 % speed, the carry move
peaks at 67 %, about 600 % and 160 % of a motor's limit respectively.

The 2D vision network, on 10,000 unseen scenes: position error 0.28 mm mean, 0.70 mm
p95, 10.4 mm max; every part found close enough to grip (the gripper catches a part
within 22 mm). With lighting and noise variation added it stays at 0.39 mm mean, with
3 in 10,000 out of reach of the gripper, all pale parts on a background lit to almost
the same colour. In the browser it finds the part to within about 0.1–0.4 mm in
about 50 ms, and in a test run Vision mode binned 30 parts with no misses other than
one caused by moving the part while the arm was reaching for it.
