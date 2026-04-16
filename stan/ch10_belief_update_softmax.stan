
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
  // -- Priors ----------------------------------------------------------
  // lr  ~ Gamma(1, 2):   exponential; E=0.5; penalises lr > 3
  // tau ~ Gamma(2, 0.5): E=4; covers plausible inverse-temperature range
  lr  ~ gamma(1, 2);
  tau ~ gamma(2, 0.5);

  // -- Forward recurrence ---------------------------------------------
  real curr_a = alpha0;
  real curr_b = beta0;
  real eps    = 1e-6;

  for (t in 1:T) {
    real kappa       = curr_a + curr_b;
    real denom_omega = kappa - 2.0;
    real omega;

    // Guard: kappa - 2 == 0 (flat Beta(1,1) prior; cannot occur with
    // alpha0=beta0=2, but retained for correctness under general inits)
    if (denom_omega <= eps) {
      omega = 0.5;
    } else {
      omega = (curr_a - 1.0) / denom_omega;
      // Clamp strictly inside (0,1) to keep logit finite
      if (omega < eps)       omega = eps;
      if (omega > 1.0 - eps) omega = 1.0 - eps;
    }

    // Softmax likelihood: eta = tau * (2*omega - 1) in (-inf, +inf)
    // bernoulli_logit_lpmf is numerically stable and vectorisable
    target += bernoulli_logit_lpmf(y[t] | tau * (2.0 * omega - 1.0));

    // Fractional pseudo-count belief update
    curr_a += lr *      y[t];
    curr_b += lr * (1 - y[t]);
  }
}

generated quantities {
  // Posterior predictive replications for PPCs after empirical fitting
  array[T] int y_rep;
  {
    real curr_a = alpha0;
    real curr_b = beta0;
    real eps    = 1e-6;

    for (t in 1:T) {
      real kappa       = curr_a + curr_b;
      real denom_omega = kappa - 2.0;
      real omega;

      if (denom_omega <= eps) {
        omega = 0.5;
      } else {
        omega = (curr_a - 1.0) / denom_omega;
        if (omega < eps)       omega = eps;
        if (omega > 1.0 - eps) omega = 1.0 - eps;
      }

      y_rep[t] = bernoulli_logit_rng(tau * (2.0 * omega - 1.0));

      curr_a += lr *      y[t];
      curr_b += lr * (1 - y[t]);
    }
  }
}

