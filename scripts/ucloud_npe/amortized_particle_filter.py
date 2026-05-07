"""Amortized Bayesian inference for the rule-based Bayesian particle filter
(Model A, Chapter 15) using BayesFlow.

This script is self-contained: it re-implements the particle filter in NumPy
(no R / reticulate / sbi dependency), trains a BayesFlow workflow on a fixed
Kruschke (1993) schedule under one of three feedback scenarios, and writes
diagnostics + posterior artifacts that the v2 chapter consumes.

Inferred parameters
-------------------
    logit_eps        : Normal(0, 1.5)            -> eps in (0, 1)
    log_n_particles  : Uniform(log 1, log 100)   -> N in {1, ..., 100}
    logit_mu         : Normal(-3, 1.5)           -> mu in (0, 1)

Scenarios (selected via --scenario)
-----------------------------------
    static            : Kruschke labels held fixed across all trials.
    contingent_shift  : labels flip after the first half of trials.
    drift             : a 1-D category boundary on `height` drifts linearly
                        from 2.0 to 3.0 across trials; feedback is the side
                        of the (possibly drifting) boundary.

Run with:
    KERAS_BACKEND=jax python amortized_particle_filter.py --scenario static
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Backend must be set BEFORE keras / bayesflow are imported.
os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bayesflow as bf

# ---------------------------------------------------------------------------
# 0. CLI + configuration
# ---------------------------------------------------------------------------

SCENARIOS = ("static", "contingent_shift", "drift")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scenario", choices=SCENARIOS, default="static",
                    help="Feedback regime (default: static).")
parser.add_argument("--n-pilot", type=int, default=20_000,
                    help="Pilot simulation budget for offline training.")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--results-root", default="particle-filter-npe",
                    help="Top-level results directory; per-scenario subdirs are created.")
parser.add_argument("--n-ppc", type=int, default=50,
                    help="Number of posterior draws used for the PPC overlay.")
parser.add_argument("--shift-streak", type=int, default=6,
                    help="contingent_shift: flip remaining labels after the "
                         "agent hits this many consecutive correct responses.")
args = parser.parse_args()

RESULTS_DIR = Path(args.results_root) / args.scenario
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 2026
N_PILOT = args.n_pilot
N_VAL = 300
N_TEST = 300
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

N_PARTICLES_MIN = 1
N_PARTICLES_MAX = 100
MAX_RULE_DIMS = 2
RESAMPLE_THRESHOLD = 0.5
N_FEATURES = 2

LOGIT_EPS_MEAN, LOGIT_EPS_SD = 0.0, 1.5
LOGIT_MU_MEAN, LOGIT_MU_SD = -3.0, 1.5
LOG_N_LOW, LOG_N_HIGH = np.log(N_PARTICLES_MIN + 1e-6), np.log(N_PARTICLES_MAX)

# ---------------------------------------------------------------------------
# 1. Kruschke (1993) stimuli + scenario-aware schedule
# ---------------------------------------------------------------------------

STIMULUS_HEIGHT = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float32)
STIMULUS_POSITION = np.array([2, 3, 1, 4, 1, 4, 2, 3], dtype=np.float32)
STIMULUS_CATEGORY = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=np.int32)
N_BLOCKS = 8

def _kruschke_order(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_stim = len(STIMULUS_HEIGHT)
    return np.concatenate([rng.permutation(n_stim) for _ in range(N_BLOCKS)])

def _drift_boundary(t: int, T: int) -> float:
    """Linearly drift the height boundary from 2.0 (t=0) to 3.0 (t=T-1)."""
    return 2.0 + (t / max(T - 1, 1)) * 1.0

def build_schedule(scenario: str, seed: int = 42):
    """Returns (height, position, base_feedback).

    For `static` and `drift`, the returned `base_feedback` is the actual
    feedback delivered on every trial (precomputed). For `contingent_shift`,
    `base_feedback` is the *pre-shift* labelling — the actual delivered
    sequence is computed online inside the particle filter, because the
    shift fires only when the agent first hits a streak of consecutive
    correct responses.
    """
    order = _kruschke_order(seed)
    height = STIMULUS_HEIGHT[order]
    position = STIMULUS_POSITION[order]
    base_feedback = STIMULUS_CATEGORY[order]
    T = height.size

    if scenario in ("static", "contingent_shift"):
        feedback = base_feedback.copy()
    elif scenario == "drift":
        feedback = np.array(
            [int(height[t] > _drift_boundary(t, T)) for t in range(T)],
            dtype=np.int32,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return height, position, feedback.astype(np.float32)

HEIGHT, POSITION, BASE_FEEDBACK = build_schedule(args.scenario)
N_TRIALS = HEIGHT.size  # 64
OBS = np.stack([HEIGHT, POSITION], axis=1)
FEATURE_RANGE = np.array([[OBS[:, 0].min(), OBS[:, 0].max()],
                          [OBS[:, 1].min(), OBS[:, 1].max()]], dtype=np.float32)

# For static/drift the feedback delivered to the agent is precomputed and
# constant across simulations. For contingent_shift it is computed online
# inside the simulator and varies across (theta, draw).
SCENARIO = args.scenario
SHIFT_STREAK = args.shift_streak

print(f"[scenario={SCENARIO}] schedule built: T={N_TRIALS}, "
      f"base feedback positives={int(BASE_FEEDBACK.sum())}/{N_TRIALS}"
      + (f"; shift fires after {SHIFT_STREAK} consecutive correct"
         if SCENARIO == "contingent_shift" else ""))

# ---------------------------------------------------------------------------
# 2. Particle filter (Model A) in NumPy
# ---------------------------------------------------------------------------

def _generate_rule(rng):
    n_dims = rng.integers(1, MAX_RULE_DIMS + 1)
    dims = rng.choice(N_FEATURES, size=n_dims, replace=False)
    thresholds = np.array(
        [rng.uniform(FEATURE_RANGE[d, 0], FEATURE_RANGE[d, 1]) for d in dims],
        dtype=np.float32,
    )
    ops = rng.integers(0, 2, size=n_dims).astype(np.int8)
    logic = rng.integers(0, 2) if n_dims == 2 else 1
    pred_if_true = rng.integers(0, 2)
    return (dims.astype(np.int8), thresholds, ops, np.int8(logic), np.int8(pred_if_true))

def _evaluate_rule(rule, stimulus):
    dims, thr, ops, logic, pred = rule
    truths = np.empty(dims.size, dtype=bool)
    for k in range(dims.size):
        v = stimulus[dims[k]]
        truths[k] = (v > thr[k]) if ops[k] == 1 else (v <= thr[k])
    rule_true = bool(truths[0]) if dims.size == 1 else (
        truths.all() if logic == 1 else truths.any()
    )
    return int(pred) if rule_true else 1 - int(pred)

def _initialize_particles(n_particles, rng):
    return [_generate_rule(rng) for _ in range(n_particles)]

def particle_filter(eps: float, n_particles: int, mu: float, rng):
    """Run Model A on the fixed schedule. Returns (responses, probs, delivered).

    `delivered` is the per-trial feedback the agent actually saw — equal to
    BASE_FEEDBACK for `static` and `drift`. For `contingent_shift` it equals
    BASE_FEEDBACK up until the agent has accumulated SHIFT_STREAK consecutive
    correct responses; from the next trial onward the labels are flipped.
    The streak counter compares response_t to the *currently active* label,
    so a flip resets the streak naturally on the trial after the shift fires.
    """
    particles = _initialize_particles(n_particles, rng)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    responses = np.empty(N_TRIALS, dtype=np.int8)
    probs = np.empty(N_TRIALS, dtype=np.float32)
    delivered = np.empty(N_TRIALS, dtype=np.float32)

    flipped = False
    streak = 0

    for t in range(N_TRIALS):
        stim = OBS[t]
        # Determine the label the agent will see on this trial.
        base = int(BASE_FEEDBACK[t])
        true_cat = (1 - base) if (SCENARIO == "contingent_shift" and flipped) else base
        delivered[t] = float(true_cat)

        preds = np.array([_evaluate_rule(p, stim) for p in particles], dtype=np.int8)
        p_cat1 = np.where(preds == 1, 1.0 - eps, eps)
        prob = float(np.clip(np.sum(weights * p_cat1), 1e-9, 1 - 1e-9))
        probs[t] = prob
        r_t = rng.binomial(1, prob)
        responses[t] = r_t

        # Streak / shift accounting (contingent_shift only).
        if SCENARIO == "contingent_shift" and not flipped:
            if int(r_t) == true_cat:
                streak += 1
                if streak >= SHIFT_STREAK:
                    flipped = True  # affects t+1 onward
            else:
                streak = 0

        # Weight update against the *delivered* label.
        likelihoods = np.where(preds == true_cat, 1.0 - eps, eps)
        new_w = weights * likelihoods
        s = new_w.sum()
        weights = new_w / s if s > 1e-12 else np.full(n_particles, 1.0 / n_particles)

        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles * RESAMPLE_THRESHOLD:
            idx = rng.choice(n_particles, size=n_particles, replace=True, p=weights)
            particles = [particles[i] for i in idx]
            weights = np.full(n_particles, 1.0 / n_particles)

        if mu > 0:
            mask = rng.random(n_particles) < mu
            for i in np.where(mask)[0]:
                particles[i] = _generate_rule(rng)

    return responses, probs, delivered

# ---------------------------------------------------------------------------
# 3. Prior + simulator
# ---------------------------------------------------------------------------

def prior():
    rng = np.random.default_rng()
    return {
        "logit_eps": np.float32(rng.normal(LOGIT_EPS_MEAN, LOGIT_EPS_SD)),
        "log_n_particles": np.float32(rng.uniform(LOG_N_LOW, LOG_N_HIGH)),
        "logit_mu": np.float32(rng.normal(LOGIT_MU_MEAN, LOGIT_MU_SD)),
    }

def observation_model(logit_eps, log_n_particles, logit_mu):
    rng = np.random.default_rng()
    eps = float(1.0 / (1.0 + np.exp(-logit_eps)))
    mu = float(1.0 / (1.0 + np.exp(-logit_mu)))
    n_part = int(np.clip(np.round(np.exp(log_n_particles)),
                         N_PARTICLES_MIN, N_PARTICLES_MAX))
    responses, _, delivered = particle_filter(eps, n_part, mu, rng)
    # The time series carries the feedback the agent actually saw — for
    # static/drift this equals BASE_FEEDBACK; for contingent_shift the shift
    # point depends on theta and the random response stream.
    series = np.stack(
        [responses.astype(np.float32), HEIGHT, POSITION, delivered],
        axis=1,
    )
    return {"trial_series": series}

simulator = bf.make_simulator([prior, observation_model])

# ---------------------------------------------------------------------------
# 4. Adapter, networks, workflow
# ---------------------------------------------------------------------------

adapter = (
    bf.Adapter()
    .as_time_series(["trial_series"])
    .convert_dtype("float64", "float32")
    .concatenate(["logit_eps", "log_n_particles", "logit_mu"],
                 into="inference_variables")
    .rename("trial_series", "summary_variables")
)

summary_net = bf.networks.TimeSeriesTransformer(summary_dim=9, time_axis=-2)
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (256, 256, 256, 256)},
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=str(RESULTS_DIR),
)

# ---------------------------------------------------------------------------
# 5. Pre-simulate pilot budget
# ---------------------------------------------------------------------------

N_TOTAL = N_PILOT + N_VAL + N_TEST
SIM_CHUNK = 200  # datasets per progress tick

print(f"\nSimulating {N_TOTAL} datasets in chunks of {SIM_CHUNK}...")
chunks = []
from tqdm import tqdm
for start in tqdm(range(0, N_TOTAL, SIM_CHUNK),
                  desc="Simulating", unit="batch",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                             "[{elapsed}<{remaining}, {rate_fmt}]"):
    n = min(SIM_CHUNK, N_TOTAL - start)
    chunks.append(workflow.simulate(n))

all_sims = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
train_data = {k: v[:N_PILOT]             for k, v in all_sims.items()}
val_data   = {k: v[N_PILOT:N_PILOT+N_VAL] for k, v in all_sims.items()}
test_data  = {k: v[N_PILOT+N_VAL:]       for k, v in all_sims.items()}
for k, v in train_data.items():
    print(f"  {k}: {v.shape}")

# ---------------------------------------------------------------------------
# 5a. Prior predictive figure (cumulative accuracy ribbon)
# ---------------------------------------------------------------------------

n_pp_show = min(500, N_PILOT)
pp_idx = np.random.default_rng(0).choice(N_PILOT, size=n_pp_show, replace=False)
pp_series = train_data["trial_series"][pp_idx]  # (n_pp, T, 4)
pp_responses = pp_series[..., 0]
pp_delivered = pp_series[..., 3]  # delivered feedback (per-sim under contingent_shift)
pp_correct = (pp_responses == pp_delivered).astype(np.float32)
pp_cumacc = np.cumsum(pp_correct, axis=1) / np.arange(1, N_TRIALS + 1)[None, :]
qs = np.quantile(pp_cumacc, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
trials = np.arange(1, N_TRIALS + 1)
ax.fill_between(trials, qs[0], qs[4], alpha=0.2, color="#56B4E9", label="5–95%")
ax.fill_between(trials, qs[1], qs[3], alpha=0.4, color="#56B4E9", label="25–75%")
ax.plot(trials, qs[2], color="#0072B2", lw=1.5, label="median")
ax.axhline(0.5, color="grey", linestyle="--", lw=0.7)
ax.set_ylim(0, 1)
ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative accuracy")
ax.set_title(f"NPE prior predictive ({args.scenario}, n={n_pp_show})")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "prior_predictive.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved prior predictive -> {RESULTS_DIR / 'prior_predictive.png'}")

# ---------------------------------------------------------------------------
# 6. Train (offline)
# ---------------------------------------------------------------------------

import keras
# Note: ReduceLROnPlateau is incompatible with BayesFlow's internal LR schedule.
# EarlyStopping alone is sufficient — BayesFlow handles LR decay internally.
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,          # stop if val_loss doesn't improve for 15 epochs
        restore_best_weights=True,
        verbose=1,
    ),
]

print(f"\nTraining offline for up to {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
print("Early stopping: patience=15 epochs on val_loss. Best weights will be restored.")
history = workflow.fit_offline(
    data=train_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_data,
    callbacks=callbacks,
)

with open(RESULTS_DIR / "history.json", "w") as f:
    json.dump(history.history, f)
print(f"Saved history -> {RESULTS_DIR / 'history.json'}")

# ---------------------------------------------------------------------------
# 7. Default diagnostics
# ---------------------------------------------------------------------------

print("\nComputing diagnostics on held-out test set...")
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics.to_csv(RESULTS_DIR / "metrics.csv")
print(metrics)

figures = workflow.plot_default_diagnostics(test_data=test_data)
figure_names = {
    "losses": "loss.png",
    "recovery": "recovery.png",
    "calibration_ecdf": "calibration_ecdf.png",
    "coverage": "coverage.png",
    "z_score_contraction": "z_score_contraction.png",
}
# BayesFlow 2.x may return a dict, a single Figure, or a list — handle all cases.
if isinstance(figures, dict):
    for key, fig in figures.items():
        out = RESULTS_DIR / figure_names.get(str(key), f"{key}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
elif hasattr(figures, "savefig"):
    figures.savefig(RESULTS_DIR / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(figures)
else:
    for i, fig in enumerate(figures):
        if hasattr(fig, "savefig"):
            fig.savefig(RESULTS_DIR / f"diagnostics_{i}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
print(f"Saved diagnostic figures -> {RESULTS_DIR}/")

# ---------------------------------------------------------------------------
# 8. Demo: amortized inference on one held-out subject
#    Emit posterior_eps.csv (raw eps draws) for the chapter's NPE-vs-Stan
#    comparison chunk.
# ---------------------------------------------------------------------------

print("\nDemo: amortized inference on one held-out simulated subject.")
demo_idx = 0
demo_obs = {"trial_series": test_data["trial_series"][demo_idx:demo_idx + 1]}
samples = workflow.sample(conditions=demo_obs, num_samples=2000)

eps_post = 1 / (1 + np.exp(-np.asarray(samples["logit_eps"]).reshape(-1)))
n_post = np.exp(np.asarray(samples["log_n_particles"]).reshape(-1))
mu_post = 1 / (1 + np.exp(-np.asarray(samples["logit_mu"]).reshape(-1)))

# Save raw eps draws + a small joint-posterior CSV for downstream consumers.
np.savetxt(RESULTS_DIR / "posterior_eps.csv",
           eps_post, header="eps", comments="", delimiter=",")
np.savetxt(
    RESULTS_DIR / "posterior_demo.csv",
    np.column_stack([eps_post, n_post, mu_post]),
    header="eps,n_particles,mu", comments="", delimiter=",",
)
print(f"Saved posterior_eps.csv + posterior_demo.csv -> {RESULTS_DIR}/")

true_eps = float(1 / (1 + np.exp(-np.asarray(test_data["logit_eps"][demo_idx]).ravel()[0])))
true_n   = float(np.exp(np.asarray(test_data["log_n_particles"][demo_idx]).ravel()[0]))
true_mu  = float(1 / (1 + np.exp(-np.asarray(test_data["logit_mu"][demo_idx]).ravel()[0])))

print(f"  eps : true={true_eps:.3f}  post mean={eps_post.mean():.3f}  "
      f"95% CI=[{np.quantile(eps_post, 0.025):.3f}, {np.quantile(eps_post, 0.975):.3f}]")
print(f"  N   : true={true_n:.1f}  post mean={n_post.mean():.1f}  "
      f"95% CI=[{np.quantile(n_post, 0.025):.1f}, {np.quantile(n_post, 0.975):.1f}]")
print(f"  mu  : true={true_mu:.3f}  post mean={mu_post.mean():.3f}  "
      f"95% CI=[{np.quantile(mu_post, 0.025):.3f}, {np.quantile(mu_post, 0.975):.3f}]")

# ---------------------------------------------------------------------------
# 9. Posterior predictive overlay
#    Skill-mandated PPC: reuse the simulator (no re-implementation),
#    loop ~50 posterior draws, overlay cumulative-accuracy curves.
# ---------------------------------------------------------------------------

n_ppc = args.n_ppc
print(f"\nRunning posterior predictive ({n_ppc} draws)...")

obs_responses = test_data["trial_series"][demo_idx, :, 0].astype(np.int32)
obs_delivered = test_data["trial_series"][demo_idx, :, 3].astype(np.int32)
obs_correct = (obs_responses == obs_delivered).astype(np.float32)
obs_cumacc = np.cumsum(obs_correct) / np.arange(1, N_TRIALS + 1)

ppc_idx = np.random.default_rng(1).choice(eps_post.size, size=n_ppc, replace=False)
ppc_curves = np.empty((n_ppc, N_TRIALS), dtype=np.float32)
for j, i in enumerate(ppc_idx):
    rng_j = np.random.default_rng(int(1_000_000 + i))
    n_j = int(np.clip(round(n_post[i]), N_PARTICLES_MIN, N_PARTICLES_MAX))
    rep, _, rep_delivered = particle_filter(
        float(eps_post[i]), n_j, float(mu_post[i]), rng_j
    )
    # Each replicate has its OWN delivered feedback under contingent_shift,
    # because the shift point depends on that replicate's response stream.
    correct = (rep == rep_delivered.astype(np.int32)).astype(np.float32)
    ppc_curves[j] = np.cumsum(correct) / np.arange(1, N_TRIALS + 1)

ppc_q = np.quantile(ppc_curves, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(trials, ppc_q[0], ppc_q[4], alpha=0.2, color="#009E73", label="PPC 5–95%")
ax.fill_between(trials, ppc_q[1], ppc_q[3], alpha=0.4, color="#009E73", label="PPC 25–75%")
ax.plot(trials, ppc_q[2], color="#006D4F", lw=1.0, label="PPC median")
ax.plot(trials, obs_cumacc, color="black", lw=1.8, label="observed (demo subject)")
ax.axhline(0.5, color="grey", linestyle="--", lw=0.7)
ax.set_ylim(0, 1)
ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative accuracy")
ax.set_title(f"Posterior predictive overlay ({args.scenario}, demo subject, n_ppc={n_ppc})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "ppc_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved PPC overlay -> {RESULTS_DIR / 'ppc_overlay.png'}")

print(f"\nDone. All artifacts in: {RESULTS_DIR.resolve()}")
print("To compare scenarios, run this script three times with --scenario "
      "static, contingent_shift, drift; results land under "
      f"{Path(args.results_root).resolve()}/<scenario>/.")
