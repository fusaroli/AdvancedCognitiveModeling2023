
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
}
parameters {
  real alpha;
}
model {
  target += normal_lpdf(alpha | 0, 1);
  target += bernoulli_logit_lpmf(h | alpha);
}
generated quantities {
  real<lower=0, upper=1> theta = inv_logit(alpha);
  vector[n] log_lik;
  array[n] int pred_choice;
  for (i in 1:n) {
    log_lik[i]    = bernoulli_logit_lpmf(h[i] | alpha);
    pred_choice[i] = bernoulli_logit_rng(alpha);
  }
}

