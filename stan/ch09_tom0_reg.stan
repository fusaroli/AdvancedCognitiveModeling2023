
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
  vector[S] ecv_z;        // standardized log ECV
  vector[S] group_z;      // standardized log group size
}
parameters {
  vector[S] mu_log_sigma;
  real<lower=0> tau_log_sigma;
  vector[J] z_log_sigma;
  // log_beta regression
  real alpha_logbeta;
  real b_ecv;
  real b_group;
  real<lower=0> tau_sp;        // species residual SD on log_beta
  vector[S] z_sp;
  real<lower=0> tau_log_beta;  // individual-within-species SD
  vector[J] z_log_beta;
  // bias
  vector[S] mu_bias;
  real<lower=0> tau_bias;
  vector[J] z_bias;
}
transformed parameters {
  vector[S] mu_log_beta = alpha_logbeta + b_ecv * ecv_z +
                          b_group * group_z + tau_sp * z_sp;
  vector[J] log_sigma = mu_log_sigma[species_of] + tau_log_sigma * z_log_sigma;
  vector[J] log_beta  = mu_log_beta[species_of]  + tau_log_beta  * z_log_beta;
  vector[J] bias      = mu_bias[species_of]      + tau_bias      * z_bias;
  vector[N] dV;
  for (j in 1:J) {
    real sigma_j = exp(log_sigma[j]);
    real mu_s    = 0.0;
    real Sig_s   = 1.0;
    for (k in start[j]:stop[j]) {
      real s_mu = inv_logit(mu_s);
      real p_op = inv_logit(mu_s / sqrt(1 + 0.36 * Sig_s));
      dV[k] = 2 * p_op - 1;
      Sig_s = 1.0 / (1.0 / (Sig_s + sigma_j) + s_mu * (1 - s_mu));
      mu_s  = mu_s + Sig_s * (op_choice[k] - s_mu);
    }
  }
}
model {
  mu_log_sigma ~ normal(-2, 1);
  tau_log_sigma ~ exponential(2);
  z_log_sigma ~ std_normal();
  alpha_logbeta ~ normal(-1, 1);
  b_ecv   ~ normal(0, 0.5);
  b_group ~ normal(0, 0.5);
  tau_sp        ~ exponential(2);
  z_sp          ~ std_normal();
  tau_log_beta  ~ exponential(1);
  z_log_beta    ~ std_normal();
  mu_bias ~ normal(0, 1);
  tau_bias ~ exponential(1);
  z_bias ~ std_normal();
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

