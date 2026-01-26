
// Stan Model: Simple Bernoulli (Logit Parameterization)
// Estimate theta on the log-odds scale

data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
}

parameters {
  real theta_logit; // Parameter is now on the unbounded log-odds scale
}

model {
  // Prior on log-odds scale (e.g., Normal(0, 1))
  // Normal(0, 1) on log-odds corresponds roughly to a diffuse prior on probability scale
  target += normal_lpdf(theta_logit | 0, 1);

  // Likelihood using the logit version of the Bernoulli PMF
  // This tells Stan that theta_logit is on the log-odds scale
  target += bernoulli_logit_lpmf(h | theta_logit);
}

generated quantities {
  // Convert estimate back to probability scale for easier interpretation
  real<lower=0, upper=1> theta = inv_logit(theta_logit);

  // Also generate prior sample on probability scale for comparison
  real<lower=0, upper=1> theta_prior = inv_logit(normal_rng(0, 1));

  // Predictions can still be generated using the probability scale theta
  array[n] int h_pred = bernoulli_rng(rep_vector(theta, n));
}

