
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
  // Prior on log-odds scale (e.g., Normal(0, 1.5))
  // Normal(0, 1.5) on log-odds corresponds roughly to a diffuse prior on probability scale
  target += normal_lpdf(theta_logit | 0, 1.5);
  // Likelihood using the logit version of the Bernoulli PMF
  // This tells Stan that theta_logit is on the log-odds scale
  target += bernoulli_logit_lpmf(h | theta_logit);
}
generated quantities {
  // Generate the prior for viz purposes
  real prior_theta = inv_logit(normal_rng(0, 1.5)); // Sample from the log-odds prior
  // Convert estimate back to probability scale for easier interpretation
  real<lower=0, upper=1> theta = inv_logit(theta_logit);
}

