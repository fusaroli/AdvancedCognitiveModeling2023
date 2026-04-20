
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0, upper=1> op_choice;
}
parameters {
  real log_sigma;
  real log_beta;
  real bias;
}
transformed parameters {
  vector[N] dV;
  {
    real sigma = exp(log_sigma);
    real mu    = 0.0;
    real Sig   = 1.0;
    for (t in 1:N) {
      real s_mu = inv_logit(mu);
      real p_op = inv_logit(mu / sqrt(1 + 0.36 * Sig));
      dV[t] = 2 * p_op - 1;
      Sig = 1.0 / (1.0 / (Sig + sigma) + s_mu * (1 - s_mu));
      mu  = mu + Sig * (op_choice[t] - s_mu);
    }
  }
}
model {
  log_sigma ~ normal(-2, 1);
  log_beta  ~ normal(-1, 1);
  bias      ~ normal( 0, 1);
  choice ~ bernoulli_logit((dV + bias) / exp(log_beta));
}
generated quantities {
  vector[N] log_lik;
  array[N] int choice_rep;
  real lprior =   normal_lpdf(log_sigma | -2, 1)
                + normal_lpdf(log_beta  | -1, 1)
                + normal_lpdf(bias      |  0, 1);
  for (t in 1:N) {
    log_lik[t]    = bernoulli_logit_lpmf(choice[t] |
                       (dV[t] + bias) / exp(log_beta));
    choice_rep[t] = bernoulli_logit_rng((dV[t] + bias) / exp(log_beta));
  }
}

