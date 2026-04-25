
data {
  int<lower=1> trials;
  int<lower=1> agents;
  array[trials, agents] int<lower=0, upper=1> h;
  // Priors as data — change without recompiling
  real prior_muA_mu;
  real<lower=0> prior_muA_sigma;
  real prior_muN_mu;
  real<lower=0> prior_muN_sigma;
  real<lower=0> prior_tau_sigma;
  real<lower=0> prior_lkj_shape;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real muA;
  real muN;
  vector<lower=0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_Omega;
}

transformed parameters {
  matrix[agents, 2] IDs = (diag_pre_multiply(tau, L_Omega) * z_IDs)';
}

model {
  target += normal_lpdf(muA | prior_muA_mu,  prior_muA_sigma);
  target += normal_lpdf(muN | prior_muN_mu,  prior_muN_sigma);
  target += normal_lpdf(tau | 0, prior_tau_sigma);
  target += lkj_corr_cholesky_lpdf(L_Omega | prior_lkj_shape);
  target += std_normal_lpdf(to_vector(z_IDs));

  for (i in 1:agents) {
    real noise_i = inv_logit(muN + IDs[i, 2]);
    real alpha_i = muA + IDs[i, 1];
    for (t in 1:trials)
      target += log_mix(noise_i,
                        bernoulli_logit_lpmf(h[t, i] | 0.0),
                        bernoulli_logit_lpmf(h[t, i] | alpha_i));
  }
}

generated quantities {
  real<lower=0, upper=1> pop_bias  = inv_logit(muA);
  real<lower=0, upper=1> pop_noise = inv_logit(muN);
  corr_matrix[2] Omega             = multiply_lower_tri_self_transpose(L_Omega);
  real bias_noise_corr             = Omega[1, 2];
  real lprior = normal_lpdf(muA | prior_muA_mu,  prior_muA_sigma)
              + normal_lpdf(muN | prior_muN_mu,  prior_muN_sigma)
              + normal_lpdf(tau | 0, prior_tau_sigma)
              + lkj_corr_cholesky_lpdf(L_Omega | prior_lkj_shape)
              + std_normal_lpdf(to_vector(z_IDs));

  array[agents] real<lower=0, upper=1> agent_bias;
  array[agents] real<lower=0, upper=1> agent_noise;
  array[trials, agents] real log_lik;

  for (i in 1:agents) {
    real alpha_i = muA + IDs[i, 1];
    real noise_i = inv_logit(muN + IDs[i, 2]);
    agent_bias[i]  = inv_logit(alpha_i);
    agent_noise[i] = noise_i;
    if (run_diagnostics) {
      for (t in 1:trials)
        log_lik[t, i] = log_mix(noise_i,
                                bernoulli_logit_lpmf(h[t, i] | 0.0),
                                bernoulli_logit_lpmf(h[t, i] | alpha_i));
    }
  }
}

