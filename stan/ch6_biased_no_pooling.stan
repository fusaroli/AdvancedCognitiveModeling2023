
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> agent;
  array[N] int<lower=0, upper=1> h;
}
parameters {
  vector[J] theta_logit; // J completely independent parameters
}
model {
  // Static prior: No hierarchical learning
  target += normal_lpdf(theta_logit | 0, 1.5);
  target += bernoulli_logit_lpmf(h | theta_logit[agent]);
}

