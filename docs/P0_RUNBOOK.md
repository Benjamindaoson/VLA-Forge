# P0: Real LIBERO / SmolVLA baseline (Linux GPU)

**Purpose:** get the *first real robot-policy simulation trace*, with a frozen,
public LIBERO-trained SmolVLA checkpoint. No GPU run has been carried out by
this repository's CPU-only CI.

## Requirements

- Linux host (Ubuntu preferred), NVIDIA GPU, compatible CUDA/PyTorch driver;
  configure a disk volume rather than using a temporary container root.
- Python 3.11-compatible environment; sufficient persistent disk space for
  weights, dataset, rendered rollouts and training caches.
- Use the pretrained checkpoint
  https://huggingface.co/lerobot/smolvla_libero for a meaningful frozen LIBERO baseline.
- Use the preprocessed dataset https://huggingface.co/datasets/lerobot/libero
  when training; record the exact revision.
- The LIBERO environment is MuJoCo-based. Do NOT install Isaac Lab or ROS2
  for this first gate.

## 1. Install dependencies

Run in an isolated Python environment; *do not change your production Python*.

~~~bash
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/Benjamindaoson/VLA-Forge.git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd lerobot
git rev-parse HEAD | tee ../lerobot_git_revision.txt
python -m pip install -e ".[smolvla,libero,training,evaluation]"
cd ../VLA-Forge
python -m pip install -e ".[dev,serve]"

export MUJOCO_GL=egl
python -m vla_forge.preflight --require-gpu
~~~

If PyTorch cannot see CUDA, stop and fix the environment before GPU work.
The source SHA, installed dependency versions, CUDA/GPU and model/dataset revisions
must be captured and pinned after a successful environment installation.
The upstream main branch is only an installation starting point, not a reproducible
long-term dependency lock.

## 2. Run minimum real-policy smoke (no training)

Start with **one task and three rollouts**, not the full four-suite benchmark.

~~~bash
export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.hard_reset=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --seed=42 \
  --output_dir=outputs/p0_smoke_object_0_seed42
~~~

This is a **proposed** LeRobot command based on the official current CLI,
and has not been executed on the user's GPU. Verify the pretrained
checkpoint's trained action parameterization (relative versus absolute),
the two camera keys, 8-dimensional state and 7-dimensional actions before
interpreting success rates. Consult the exact pinned LeRobot --help/config
because flags can change across revisions.

Smoke acceptance:
1. A CUDA model loads and produces finite actions.
2. LIBERO creates an actual environment and executes all requested episodes.
3. The output contains task/episode success flags, execution logs, configuration,
   policy ID, simulator revision and evidence or captures sufficient for debugging.
4. No exception from image preprocessing, action normalization or shape mismatch.
5. Save GPU VRAM high-water mark, wall time and disk usage. Success rate is
   **not** a training improvement yet.

## 3. Freeze baseline protocol

After the smoke is stable, run **40 tasks x 10 episodes = 400 rollouts**
over Spatial/Object/Goal/Long. This is a baseline, with *no model updates*.

~~~bash
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.hard_reset=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=42 \
  --output_dir=outputs/p0_baseline_seed42
~~~

For stronger measurements, repeat the *same frozen protocol* with 3 seed
sets and report task-level variance. If comparing two checkpoints with
paired states, document actual reset state IDs and validate LeRobot batch/reset
behavior; same seed is not automatically proof of identical initial states.
The docs suggest running each task in a single batch for matching;
this may raise memory demand, so validate actual state identity carefully.

## 4. Diagnose timing vs model skill without training

At a few locked physical initial states:
- Confirm camera frame order, 8D state normalization, 7D action control.
- Measure inference latency, configured action-horizon execution and effective
  control rate.
- Compare two *predeclared* action-chunk execution lengths while keeping
  checkpoint and initial states matched.
- Save raw failure episodes and link policy, task, scene and model version.
- Start implementing the real LIBERO SimulatorAdapter/restore interface from
  Issue #1. First prove replay fidelity before interpreting counterfactuals.

Do not label a runtime-only change as a VLA model-weight improvement.

## 5. When training is permitted

- **Technical SFT smoke:** after valid dataset loading and baseline smoke,
  run 20-100 steps with batch size 1-2, save the checkpoint and run a finite
  forward/backward test. The objective is that the stack works, *not* better skill.
- **Actual corrective fine-tune:** only when verified corrective episodes exist,
  lineage-family train/test splits are frozen, and the training bridge from
  verified labels to actual SmolVLA Flow Matching is tested.
- Start from the same frozen baseline and compare random extra demos,
  corrective SFT and replay under matched expert/compute budgets.

There is no real corrected dataset in this repo today. The weighted FM helper
is not yet connected to LeRobot training, so an ordinary lerobot-train command
does **not** automatically implement BIR or failure-weighted training.

## P0 deliverable and open blockers

Upload a run manifest containing: Linux/CUDA, dependency revisions, source and
model commit hash, fixed scenario/seed IDs, config, rollout/video paths, first
success/failure list, GPU VRAM, wall time, error and retry records.

Do not start full Isaac Lab/PPO, ROS2 integration, TensorRT conversion or
expensive VLA post-training before this gate has real evidence.

### Official references

- https://huggingface.co/docs/lerobot/libero
- https://huggingface.co/docs/lerobot/smolvla
- https://huggingface.co/lerobot/smolvla_libero
- https://huggingface.co/datasets/lerobot/libero
