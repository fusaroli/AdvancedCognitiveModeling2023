
functions {
  real jardri_f(real L, real w) {
    real num = w * exp(L) + (1.0 - w);
    real den = (1.0 - w) * exp(L) + w;
    return log(num / den);
  }
}
data {
  int<lower=1>                   N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0>          k_self;
  array[N] int<lower=0>          n_self;
  array[N] int<lower=0>          k_social;
  array[N] int<lower=0>          n_social;
  int<lower=0, upper=1>          run_diagnostics;
}
transformed data {
  real a0 = 0.5;
  real b0 = 0.5;
  array[N] real lo_self;
  array[N] real lo_social;
  for (i in 1:N) {
    lo_self[i]   = log((a0 + k_self[i])   / (b0 + n_self[i]   - k_self[i]));
    lo_social[i] = log((a0 + k_social[i]) / (b0 + n_social[i] - k_social[i]));
  }
}
parameters {
  real<lower=0.5, upper=1.0> w_self;
  real<lower=0.5, upper=1.0> w_social;
  real<lower=0>              alpha_self_m1;    // excess over 1
  real<lower=0>              alpha_social_m1;
}
transformed parameters {
  real<lower=1> alpha_self   = 1.0 + alpha_self_m1;
  real<lower=1> alpha_social = 1.0 + alpha_social_m1;
}
model {
  w_self   ~ normal(0.75, 0.25);
  w_social ~ normal(0.75, 0.25);
  alpha_self_m1   ~ std_normal();
  alpha_social_m1 ~ std_normal();
  for (i in 1:N)
    target += bernoulli_logit_lpmf(y[i] |
                jardri_f(alpha_self   * lo_self[i],   w_self) +
                jardri_f(alpha_social * lo_social[i], w_social));
}
generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  if (run_diagnostics) {
    for (i in 1:N) {
      real L = jardri_f(alpha_self   * lo_self[i],   w_self) +
               jardri_f(alpha_social * lo_social[i], w_social);
      log_lik[i] = bernoulli_logit_lpmf(y[i] | L);
      y_rep[i]   = bernoulli_logit_rng(L);
    }
  }
}

