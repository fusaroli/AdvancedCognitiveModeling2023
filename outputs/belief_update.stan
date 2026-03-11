
data {
  int<lower=1>                    T;
  array[T] int<lower=0, upper=1>  y;
  real<lower=0>                   alpha0;
  real<lower=0>                   beta0;
}

parameters {
  real<lower=0> lr;
  real<lower=0> tau;
}

model {
  // -- Priors --------------------------------------------------------
  // lr  ~ Gamma(1, 2):  exponential with mean 0.5; penalises lr > 2
  // tau ~ Gamma(8, 1):  mean 8, mode 7; prevents near-flat predictions
  lr  ~ gamma(1, 2);
  tau ~ gamma(8, 1);

  // -- Forward recurrence -------------------------------------------
  real curr_a      = alpha0;
  real curr_b      = beta0;
  real eps         = 1e-6;

  for (t in 1:T) {
    real kappa       = curr_a + curr_b;
    real denom_omega = kappa - 2.0;
    real omega;
    real bp_t;
    real ap_t;
    real denom_ap;

    // Guard: kappa - 2 == 0 (flat Beta(1,1) prior edge case)
    if (denom_omega <= eps) {
      omega = 0.5;
    } else {
      omega = (curr_a - 1.0) / denom_omega;
      if (omega < eps)       omega = eps;
      if (omega > 1.0 - eps) omega = 1.0 - eps;
    }

    // Clamp bp_t first, then re-derive ap_t from the clamped value
    // so the pair (ap_t, bp_t) remains algebraically consistent.
    bp_t = tau * kappa * (1.0 - omega) + 2.0 * omega - 1.0;
    if (bp_t < 0.01) bp_t = 0.01;

    // Guard: 1 - omega == 0 (long success streaks drive omega -> 1)
    denom_ap = 1.0 - omega;
    if (denom_ap <= eps) {
      ap_t = bp_t;
    } else {
      ap_t = (omega * bp_t - 2.0 * omega + 1.0) / denom_ap;
      if (ap_t < 0.01) ap_t = 0.01;
    }

    // Marginal Bernoulli: integrates out p_t ~ Beta(ap_t, bp_t)
    target += bernoulli_lpmf(y[t] | ap_t / (ap_t + bp_t));

    // Fractional pseudo-count update
    curr_a += lr *      y[t];
    curr_b += lr * (1 - y[t]);
  }
}

