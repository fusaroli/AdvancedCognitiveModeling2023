
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> S;
  array[J] int<lower=1, upper=S> species_of;
  array[N] int<lower=1, upper=J> agent;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0, upper=1> op_choice;
  array[J] int<lower=1> start;
  array[J] int<lower=1> stop;
}
parameters {
  vector[S] mu_log_sigma_op;
  real<lower=0> tau_log_sigma_op;
  vector[J] z_log_sigma_op;
  vector[S] mu_log_beta_op;
  real<lower=0> tau_log_beta_op;
  vector[J] z_log_beta_op;
  vector[S] mu_log_beta;
  real<lower=0> tau_log_beta;
  vector[J] z_log_beta;
  vector[S] mu_bias;
  real<lower=0> tau_bias;
  vector[J] z_bias;
}
transformed parameters {
  vector[J] log_sigma_op = mu_log_sigma_op[species_of] + tau_log_sigma_op * z_log_sigma_op;
  vector[J] log_beta_op  = mu_log_beta_op[species_of]  + tau_log_beta_op  * z_log_beta_op;
  vector[J] log_beta     = mu_log_beta[species_of]     + tau_log_beta     * z_log_beta;
  vector[J] bias         = mu_bias[species_of]         + tau_bias         * z_bias;
  vector[N] dV;
  for (j in 1:J) {
    real sigma_op_j = exp(log_sigma_op[j]);
    real beta_op_j  = exp(log_beta_op[j]);
    real mu_s  = 0.0;
    real Sig_s = 1.0;
    for (k in start[j]:stop[j]) {
      real p_self_from_op = inv_logit(mu_s / sqrt(1 + 0.36 * Sig_s));
      real p_op           = inv_logit((1 - 2 * p_self_from_op) / beta_op_j);
      dV[k] = 2 * p_op - 1;
      real s_ps = p_self_from_op;
      Sig_s = 1.0 / (1.0 / (Sig_s + sigma_op_j) + s_ps * (1 - s_ps));
      mu_s  = mu_s + Sig_s * (choice[k] - s_ps);
    }
  }
}
model {
  mu_log_sigma_op  ~ normal(-2, 1);
  tau_log_sigma_op ~ exponential(2);
  z_log_sigma_op   ~ std_normal();
  mu_log_beta_op   ~ normal(-1, 0.7);
  tau_log_beta_op  ~ exponential(2);
  z_log_beta_op    ~ std_normal();
  mu_log_beta      ~ normal(-1, 1);
  tau_log_beta     ~ exponential(1);
  z_log_beta       ~ std_normal();
  mu_bias          ~ normal(0, 1);
  tau_bias         ~ exponential(1);
  z_bias           ~ std_normal();
  for (n in 1:N)
    choice[n] ~ bernoulli_logit(
      (dV[n] + bias[agent[n]]) / exp(log_beta[agent[n]])
    );
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(
      choice[n] | (dV[n] + bias[agent[n]]) / exp(log_beta[agent[n]])
    );
}

