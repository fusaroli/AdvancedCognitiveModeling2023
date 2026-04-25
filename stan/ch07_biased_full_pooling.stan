
data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
}
parameters {
  real theta_logit; // A single parameter for the entire universe
}
model {
  target += normal_lpdf(theta_logit | 0, 1.5);
  target += bernoulli_logit_lpmf(y | theta_logit);
}

