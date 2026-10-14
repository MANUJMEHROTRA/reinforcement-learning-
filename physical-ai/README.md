# Physical AI from First Principles

A hands-on course on **Physical AI** — training policies that act in the physical world — built for someone who already knows reinforcement learning.

Every notebook starts from real data or real physics, implements the idea **from scratch** (`numpy`/`pytorch` only), and only switches to production libraries (`lerobot`, `mujoco` high-level APIs) in the last two notebooks. Nothing is hidden behind a library call until you've already built it yourself.

This is the companion course to [`reinforcement-learning`](../reinforcement-learning). It assumes you've done that one.

---

## Part 0 — Top down: what does this actually get you?

### The capability you are building toward

A function. It takes what the robot can see and feel — camera frames plus joint positions — and outputs motor commands, 30 times a second:

```
π(camera_images, joint_positions) → target_joint_positions
```

That's it. That single function, trained well, is what makes a robot pick up a cup it has never seen, fold a towel, or plug in a cable. Everything in this course is about how to get that function.

### Why this is suddenly worth learning

Three things changed, and they changed recently:

1. **Hardware collapsed in price.** The SO-101 arm is a 6-axis, 3D-printable robot costing **$100–500**. It runs the *same policies* that academic labs run on $30,000 industrial arms.
2. **Data became shareable.** HuggingFace now hosts thousands of real robot demonstration datasets in one standard format. You can train on someone else's robot before you own one.
3. **The method stabilized.** The field converged on a recipe: *imitation learning on demonstrations, then RL fine-tuning*. That recipe is learnable in months, not years.

### Where this shows up in the real world

| Setting | What the policy does | Who's doing it |
|---|---|---|
| Warehouse picking | Grasp arbitrary items from a bin | Amazon Robotics, Covariant |
| Lab automation | Pipetting, sample handling, plate moving | pharma, academic labs |
| Assembly & manufacturing | Insert connectors, route cables, screw fasteners | Tesla, Figure, Agility |
| Household tasks | Laundry folding, table clearing, dishwasher loading | Physical Intelligence (π0) |
| Humanoid locomotion | Walking, balancing, recovering from shoves | Unitree, Boston Dynamics, NVIDIA GR00T |

The thing worth noticing: **laundry folding and bin picking are the same problem** at the level of the function above. That's why foundation models for robots work at all.

### What you'll personally be able to do at the end

- Load and interrogate any robot dataset on HuggingFace, and know whether it's any good
- Build a physics simulator from scratch, then use a real one properly
- Write a controller before you try to learn one, so you know what learning is replacing
- Train a behavior-cloning policy and a diffusion policy on real robot data
- Train a locomotion policy with RL in simulation
- Measure your own sim-to-real gap instead of guessing at it
- Deploy to a physical arm, if you buy one

### The honest caveat, up front

**Pure RL from scratch on a real robot does not work in 2026.** Sample complexity, hardware wear, and safety rule it out. RL's roles in physical AI are narrow and specific:

- RL **in simulation** for locomotion — this works extremely well
- RL **fine-tuning** of a policy that imitation learning already got most of the way
- RL for **sim-to-real** robustness via domain randomization

If you study physical AI expecting RL to be the main character, you'll learn the wrong things well. Imitation learning is the main character. RL is the specialist you call in at the end. This course is ordered accordingly — and notebook 1 shows you the measurement that explains *why*.

---

## Part 1 — What transfers from your RL work, and what breaks

You know multi-turn agentic RL. A robot episode is also multi-turn. Here's the honest mapping:

| | Agentic RL (you know this) | Physical AI (this course) |
|---|---|---|
| Formalism | POMDP | POMDP — **same** |
| One "turn" | a message, ~100 tokens | one control step, **33 ms** |
| Episode length | 5–20 turns | **300+ steps** |
| Action | a token sequence | 6 continuous joint targets |
| Observation | tool output text | camera frames + joint encoders |
| Reward | verifier, 0/1 | hand-designed, or human judgement |
| **Can you reset?** | **yes** — rerun the prompt | **no** — time only moves forward |
| **Can you branch?** | **yes** — resample from any prefix | **no** — the cup is already knocked over |
| Cost of a bad action | wasted tokens | **broken hardware** |

The two bolded rows are the whole difference. Every branching credit-assignment method you know — rollout trees, sibling comparison, VinePPO-style resampling — assumes you can restore an intermediate state. A physical robot cannot. That single constraint is why this field looks so different from LLM RL, and why imitation learning dominates it.

Your genuine advantages coming in: you already understand policy gradients, advantage estimation, reward hacking, and credit assignment over long horizons. Most people entering robot learning come from vision or controls and find all of that unfamiliar.

Your genuine gaps: rigid-body dynamics, classical control, off-policy continuous control (SAC/TD3 — you have on-policy methods), and the fact that your environment is made of atoms.

---

## Notebook roadmap

### Part I — The data (understand it before you model it)

**1. [`01_what_is_physical_ai.ipynb`](notebooks/01_what_is_physical_ai.ipynb)**
Load a real SO-101 teleoperation dataset from HuggingFace into pandas. Then the central experiment: discover empirically *what a robot action actually is*. The answer — a position target the arm chases with lag — is the single fact that explains most of this course.

**2. `02_anatomy_of_a_demonstration.ipynb`**
Deep dive into the data in pandas. Episodes, joints, velocities, phase structure of a grasp. Where demonstrations are inconsistent and why that matters. Build the dataset statistics you'll normalize with later.

### Part II — Simulation & system identification ✅ *built and verified*

**The job this part maps onto.** A representative robot-learning posting asks you to
*"bridge the sim-to-real gap using **system identification**, domain randomization, and
actuator/dynamics modeling"* and lists MuJoCo plus URDF/MJCF as hard requirements. Part II
is that task, end to end, on real hardware data. India alone had **9,000+ open physical-AI
roles** as of mid-2026, and the reported bottleneck is breadth — people who can combine
robotics, ML and embedded work rather than one of the three.

**3. [`03_physics_from_scratch.ipynb`](notebooks/03_physics_from_scratch.ipynb)**
Write a rigid-body integrator yourself. Explicit Euler gains **+28.6%** energy in 10 s and
will eventually destroy any simulation; reordering two lines fixes it (+0.06%). Then
validate against MuJoCo — two independent implementations agreeing to **0.19°**. Ends on
the PD-controller model of a servo, and why stiffness and actuation lag are two views of
one number.

**4. [`04_the_robot_model.ipynb`](notebooks/04_the_robot_model.ipynb)**
Load the **official DeepMind Menagerie SO-101 model** — the same arm the dataset came from.
Forward kinematics from scratch (Rodrigues + homogeneous transforms), validated to
**sub-micron** agreement. Then read the model like a spec you're accountable for, and find
that its `kp` is *derived from an assumption* ("servo proportional gain set to 16") and its
backlash is modelled but disabled. Includes catching our own experiment being invalid: the
actuator saturates at a **0.17° error**, so a step test measures the force limit, not the gain.

**5. [`05_system_identification.ipynb`](notebooks/05_system_identification.ipynb)** ← the portfolio notebook
Replay the real recorded commands through the model, discover it tracks in **0 frames where
the real arm takes 4**, and fit the parameters until it matches. Held-out results:

| | lag (frames) | RMSE |
|---|---|---|
| Official Menagerie model | 0 | 6.19 |
| **Identified** | **4** | **1.44** |
| Real arm (target) | 4 | — |

**77% RMSE reduction on held-out episodes**, with the lag matching exactly. Validated
against an *independent* statistic never used in the objective — tracking-error-vs-speed
correlation, real 0.885 vs identified 0.977 — and the small overshoot is read honestly as
evidence of unmodelled backlash/stiction. Ships `artifacts/so101_identified.json` (with a
`parameter_confidence` section flagging which parameters are *weakly* identified) and a
loadable `models/so101/so101_identified.xml`.

### Part III — Learning policies ✅ *notebooks 6-7 built and verified*

**The job this part maps onto.** Postings name *"imitation-learning methods such as
diffusion policies and ACT"* and *"RL fine-tuning of pretrained VLAs"*. Part III does
imitation first (because teleop data has no reward), then RL. Crucially, it uses
**Part II's identified simulator as the evaluation harness** — which is what makes the
central finding below measurable at all.

**6. [`06_behavior_cloning.ipynb`](notebooks/06_behavior_cloning.ipynb)**
Train BC on the real SO-101 data with notebook 2's normalizers. Held-out action MAE is
**1.6 units** — looks shippable. Then close the loop in the identified simulator:

| metric | value |
|---|---|
| teacher-forced MAE (what papers report) | **1.65 units** |
| closed-loop drift @ 30 frames | 6.1 |
| closed-loop drift @ end of episode | **35.4** |

Diagnosed as **phase ambiguity**, not multimodality: the arm revisits poses ~6 s apart
that demand opposite actions, 8-19 times per episode. Includes a prediction I got wrong —
I expected the policy to freeze from label averaging; it moves at **97% of human speed**
and goes the wrong way instead, which is a more dangerous failure and ruled out an entire
family of fixes.

**7. [`07_three_fixes.ipynb`](notebooks/07_three_fixes.ipynb)** ← the headline result
Predictions written down *first*, then action chunking, vision and history conditioning
each measured both offline and closed-loop. Two of three predictions failed. The finding
that outranks all of them:

> **`corr(offline MSE, closed-loop drift) = −0.25`** across 12 variants. The best policy
> (history + chunking, drift 20.15) has **5.6× worse validation loss** than the model
> validation loss prefers. Selecting on the offline metric costs 12% of the achievable
> improvement.

Action chunking **tripled** offline loss while cutting drift 23% — which is why a field
selecting on validation loss would never have found it. Vision gave a marginal −7% offline
for 40× the training time and cannot be evaluated closed-loop at all, since our simulator
has no object.

**8. `08_rl_in_simulation.ipynb`** *(in progress)*
SAC from scratch — twin critics, tanh-Gaussian actor with the squashing correction,
reparameterised gradient, learned entropy temperature, replay buffer — filling the
off-policy continuous-control gap in an on-policy RL background. Trained in the identified
simulator, with a reward-hacking ablation and BC→RL fine-tuning.

### Part IV — Reality and production ✅ *built and verified*

**9. [`09_domain_randomization.ipynb`](notebooks/09_domain_randomization.ipynb)**
Planned as "how wide should the randomization be." The measurements turned it into "when
should you randomize at all."

| perturbation | worst case vs nominal |
|---|---|
| **delay** | **+25.4 cm** |
| armature | +1.3 cm |
| `kp` (13× range!) | +0.1 cm |
| damping | +0.0 cm |

**Actuator parameters barely matter** for a closed-loop position-controlled policy —
feedback absorbs gain error. **Latency is the entire sim-to-real gap**, because the
observation that would correct a delayed command describes a world where it hasn't landed.
Then the uncomfortable part: **domain randomization produced a policy worse than random**
(32.1 cm vs 23.2 cm), and my notebook-5 claim that "randomizing around a wrong centre
doesn't help" was **refuted** — the official-centred run beat the identified-centred one.
The failure localises to DR's two real preconditions: the randomized axis must *matter*,
and the policy must be able to *infer* which world it's in. A single-frame policy cannot
infer its own latency, so it can only hedge.

**10. [`10_production_deployment.ipynb`](notebooks/10_production_deployment.ipynb)**
Three artifacts other people and machines can consume:

- **A LeRobot v3 dataset written from scratch** — 21/21 schema checks pass, and
  notebook 1's lag analysis runs on it *unmodified* and recovers the 4-frame lag we
  injected. Semantic round-trip, not just structural.
- **A self-contained policy export** — weights, observation contract, both normalizer
  sets, action semantics, control rate, the identified dynamics it assumes, and an
  explicit limitations list. Verified by running its reference implementation in a fresh
  subprocess.
- **A 10-item deployment checklist**, every item traceable to a specific failure earlier
  in the course. One item fails on purpose: `scene_box.xml` places its box at x = 0.5 m
  against a 0.405 m reach.

Deliberately does **not** install `lerobot` — it pins torch 2.10 against our 2.14 and
would break notebooks 3–9. The separate-environment commands are in §5; the two
environments exchange files, which is what a standard format is for.

---

## The through-line

Every notebook's useful result came from a measurement contradicting something plausible.

| nb | the plausible thing | what measurement said |
|---|---|---|
| 1 | an action is a torque | position target, 4-frame lag, **50/50 episodes** |
| 2 | robot data is multimodal | **not this dataset** — 0.0% of gaps exceed baseline |
| 4 | a 30° step measures stiffness | saturates at **0.17°**, measuring the force limit |
| 5 | the official model describes the robot | 0-frame lag vs real 4; RMSE **6.19 → 1.44** |
| 6 | BC averaging makes the policy freeze | moves at **97% speed**, goes the wrong way |
| 7 | chunking helps via multimodality | helps via feedback rate; offline metrics rank it **−0.25** |
| 8 | SAC needed more samples | two bugs; **29.6 → 2.0 cm** |
| 9 | DR buys robustness for conservatism | **worse on both axes**; latency is the only axis that matters |

Eight notebooks, eight corrections, all left visible. The corrections are the
transferable part; the specific numbers are not.

## Setup

```bash
cd physical-ai
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name physical-ai
```

Notebooks 1–9 run on CPU, including Apple Silicon. MuJoCo on an M-series Mac is genuinely fast — MJX reaches roughly 650k steps/s on an M3 Max. You do not need an NVIDIA GPU for this course. Notebook 10 is lighter if you have one, but works without.

Verified working on macOS (Apple Silicon), Python 3.11, `mujoco` 3.15, `gymnasium` 1.4.

## Datasets used

| Dataset | What it is | Size |
|---|---|---|
| [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | Real SO-101 arm, teleoperated pick-and-place, 6 joints, 2 cameras, 30 fps | 50 episodes, 11,939 frames |
| [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) | Planar pushing, 2-D state and action, **includes rewards** so it maps onto an MDP directly | 206 episodes, 25,650 frames |

## Robot models used

| Model | Source | Notes |
|---|---|---|
| `models/so101/so101.xml` | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101` | The official SO-101 model. Joint names match the dataset exactly. |
| `models/so101/so101_identified.xml` | produced by notebook 5 | Same model with actuator parameters fitted to the real data. |

Both are in LeRobot v3 format: Parquet for low-dimensional signals, MP4 for video. You'll read the Parquet straight into pandas, because understanding your data is not an optional step.
