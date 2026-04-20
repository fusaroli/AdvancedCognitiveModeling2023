
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
  vector[S] ecv_z;
  vector[S] group_z;
}
parameters {
  // 0-ToM-only
  vector[S] mu_log_sigma;
  real<lower=0> tau_log_sigma;
  vector[J] z_log_sigma;
  // 1-ToM-only
  vector[S] mu_log_sigma_op;
  real<lower=0> tau_log_sigma_op;
  vector[J] z_log_sigma_op;
  vector[S] mu_log_beta_op;
  real<lower=0> tau_log_beta_op;
  vector[J] z_log_beta_op;
  // Shared decision layer
  vector[S] mu_log_beta;
  real<lower=0> tau_log_beta;
  vector[J] z_log_beta;
  vector[S] mu_bias;
  real<lower=0> tau_bias;
  vector[J] z_bias;
  // Mixture regression on architecture probability
  real alpha_pi;
  real b_ecv;
  real b_group;
  real<lower=0> tau_sp_pi;
  vector[S] u_sp;
}
transformed parameters {
  vector[J] log_sigma    = mu_log_sigma[species_of]    + tau_log_sigma    * z_log_sigma;
  vector[J] log_sigma_op = mu_log_sigma_op[species_of] + tau_log_sigma_op * z_log_sigma_op;
  vector[J] log_beta_op  = mu_log_beta_op[species_of]  + tau_log_beta_op  * z_log_beta_op;
  vector[J] log_beta     = mu_log_beta[species_of]     + tau_log_beta     * z_log_beta;
  vector[J] bias_j       = mu_bias[species_of]         + tau_bias         * z_bias;
  vector[S] logit_pi     = alpha_pi + b_ecv * ecv_z + b_group * group_z
                           + tau_sp_pi * u_sp;
  vector[S] pi_s         = inv_logit(logit_pi);
  vector[N] dV0;
  vector[N] dV1;
  for (j in 1:J) {
    // 0-ToM recursion: belief about opponent bias, updated by op_choice
    {
      real sig0 = exp(log_sigma[j]);
      real mu   = 0.0;
      real Sig  = 1.0;
      for (k in start[j]:stop[j]) {
        real s_mu   = inv_logit(mu);
        real p_op_0 = inv_logit(mu / sqrt(1 + 0.36 * Sig));
        dV0[k] = 2 * p_op_0 - 1;
        Sig = 1.0 / (1.0 / (Sig + sig0) + s_mu * (1 - s_mu));
        mu  = mu + Sig * (op_choice[k] - s_mu);
      }
    }
    // 1-ToM recursion: belief about opponent's belief about self,
    // updated by own choice
    {
      real sig_op = exp(log_sigma_op[j]);
      real b_op   = exp(log_beta_op[j]);
      real mu_s   = 0.0;
      real Sig_s  = 1.0;
      for (k in start[j]:stop[j]) {
        real p_self_from_op = inv_logit(mu_s / sqrt(1 + 0.36 * Sig_s));
        real p_op_1         = inv_logit((1 - 2 * p_self_from_op) / b_op);
        dV1[k] = 2 * p_op_1 - 1;
        real s_ps = p_self_from_op;
        Sig_s = 1.0 / (1.0 / (Sig_s + sig_op) + s_ps * (1 - s_ps));
        mu_s  = mu_s + Sig_s * (choice[k] - s_ps);
      }
    }
  }
}
model {
  // Hierarchical priors
  mu_log_sigma    ~ normal(-2, 1);
  tau_log_sigma   ~ exponential(2);
  z_log_sigma     ~ std_normal();
  mu_log_sigma_op ~ normal(-2, 1);
  tau_log_sigma_op ~ exponential(2);
  z_log_sigma_op  ~ std_normal();
  mu_log_beta_op  ~ normal(-1, 0.7);
  tau_log_beta_op ~ exponential(2);
  z_log_beta_op   ~ std_normal();
  mu_log_beta     ~ normal(-1, 1);
  tau_log_beta    ~ exponential(1);
  z_log_beta      ~ std_normal();
  mu_bias         ~ normal(0, 1);
  tau_bias        ~ exponential(1);
  z_bias          ~ std_normal();
  // Mixture regression
  alpha_pi  ~ normal(0, 1);
  b_ecv     ~ normal(0, 0.7);
  b_group   ~ normal(0, 0.7);
  tau_sp_pi ~ exponential(1);
  u_sp      ~ std_normal();
  // Marginalized likelihood: one log_mix per individual
  for (j in 1:J) {
    real ll0 = 0;
    real ll1 = 0;
    real inv_b = 1.0 / exp(log_beta[j]);
    for (k in start[j]:stop[j]) {
      ll0 += bernoulli_logit_lpmf(choice[k] |
                (dV0[k] + bias_j[j]) * inv_b);
      ll1 += bernoulli_logit_lpmf(choice[k] |
                (dV1[k] + bias_j[j]) * inv_b);
    }
    target += log_mix(pi_s[species_of[j]], ll1, ll0);
  }
}
generated quantities {
  // Per-individual posterior probability z_j = 1 (= 1-ToM)
  vector[J] post_pi_j;
  // Per-individual total log-likelihood difference (1-ToM - 0-ToM).
  vector[J] ll_diff_j;
  // Per-trial log_lik under the marginal mixture for PSIS-LOO
  vector[N] log_lik;
  for (j in 1:J) {
    real ll0 = 0;
    real ll1 = 0;
    real inv_b = 1.0 / exp(log_beta[j]);
    for (k in start[j]:stop[j]) {
      ll0 += bernoulli_logit_lpmf(choice[k] |
                (dV0[k] + bias_j[j]) * inv_b);
      ll1 += bernoulli_logit_lpmf(choice[k] |
                (dV1[k] + bias_j[j]) * inv_b);
    }
    ll_diff_j[j] = ll1 - ll0;
    real lp0 = log1m(pi_s[species_of[j]]) + ll0;
    real lp1 = log(pi_s[species_of[j]])   + ll1;
    post_pi_j[j] = exp(lp1 - log_sum_exp(lp0, lp1));
    for (k in start[j]:stop[j]) {
      real l0 = bernoulli_logit_lpmf(choice[k] |
                  (dV0[k] + bias_j[j]) * inv_b);
      real l1 = bernoulli_logit_lpmf(choice[k] |
                  (dV1[k] + bias_j[j]) * inv_b);
      log_lik[k] = log_mix(pi_s[species_of[j]], l1, l0);
    }
  }
}

