
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0, upper=1> op_choice;
}
parameters {
  real log_sigma_op;
  real log_beta_op;
  real log_beta;
  real bias;
}
transformed parameters {
  vector[N] dV;
  {
    real sigma_op = exp(log_sigma_op);
    real beta_op  = exp(log_beta_op);
    real mu_s     = 0.0;
    real Sig_s    = 1.0;
    for (t in 1:N) {
      real p_self_from_op = inv_logit(mu_s / sqrt(1 + 0.36 * Sig_s));
      real p_op           = inv_logit((1 - 2 * p_self_from_op) / beta_op);
      dV[t] = 2 * p_op - 1;
      // After observing self's choice, opponent updates its belief
      real s_ps = p_self_from_op;
      Sig_s = 1.0 / (1.0 / (Sig_s + sigma_op) + s_ps * (1 - s_ps));
      mu_s  = mu_s + Sig_s * (choice[t] - s_ps);
    }
  }
}
model {
  log_sigma_op ~ normal(-2, 1);
  log_beta_op  ~ normal(-1, 0.7);
  log_beta     ~ normal(-1, 1);
  bias         ~ normal( 0, 1);
  choice ~ bernoulli_logit((dV + bias) / exp(log_beta));
}
generated quantities {
  vector[N] log_lik;
  real lprior =   normal_lpdf(log_sigma_op | -2, 1)
                + normal_lpdf(log_beta_op  | -1, 0.7)
                + normal_lpdf(log_beta     | -1, 1)
                + normal_lpdf(bias         |  0, 1);
  for (t in 1:N)
    log_lik[t] = bernoulli_logit_lpmf(choice[t] |
                  (dV[t] + bias) / exp(log_beta));
}

