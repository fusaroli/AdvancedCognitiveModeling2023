
data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  array[N_total] int<lower=1, upper=N_subjects> subj_id;
  array[N_total] int<lower=0, upper=1> y;
}
parameters {
  vector[N_subjects] theta_logit; // N_subjects completely independent parameters
}
model {
  // Static prior: No hierarchical learning
  target += normal_lpdf(theta_logit | 0, 1.5);
  target += bernoulli_logit_lpmf(y | theta_logit[subj_id]);
}

