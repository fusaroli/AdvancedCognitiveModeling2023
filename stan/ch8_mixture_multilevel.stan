
data {
  int<lower=1> trials;
  int<lower=1> agents;
  array[trials, agents] int<lower=0, upper=1> h;
}

parameters {
  real muA;                         // population mean bias (logit)
  real muN;                         // population mean noise (logit)
  vector<lower=0>[2] tau;           // population SDs: [bias, noise]
  matrix[2, agents] z_IDs;          // standardised individual offsets
  cholesky_factor_corr[2] L_Omega;  // Cholesky of correlation matrix
}

transformed parameters {
  // Non-centred: individual offsets on the logit scale
  matrix[agents, 2] IDs = (diag_pre_multiply(tau, L_Omega) * z_IDs)';
}

model {
  target += normal_lpdf(muA | 0, 1);
  target += normal_lpdf(muN | -1, 0.5);

  // tau is declared vector<lower=0>[2], so Stan applies the log Jacobian
  // of the log transform automatically. The truncation normalising constant
  // is already absorbed into the unconstrained parameterisation; manually
  // subtracting normal_lccdf() would add unnecessary AD nodes for a
  // constant that cancels. Vectorised call covers both SD elements.
  target += normal_lpdf(tau | 0, 0.3);

  target += lkj_corr_cholesky_lpdf(L_Omega | 2);
  target += std_normal_lpdf(to_vector(z_IDs));

  // Per-agent, per-trial mixture likelihood (correct loop structure).
  // Both components use bernoulli_logit_lpmf for scale consistency:
  // logit(0.5) = 0.0, so the random component is bernoulli_logit_lpmf(h | 0.0).
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
  // Population-level summaries
  real<lower=0, upper=1> pop_bias  = inv_logit(muA);
  real<lower=0, upper=1> pop_noise = inv_logit(muN);
  corr_matrix[2] Omega             = multiply_lower_tri_self_transpose(L_Omega);
  real bias_noise_corr             = Omega[1, 2];

  // lprior is required by priorsense::powerscale_sensitivity().
  // Must mirror every term in the model block exactly.
  real lprior = normal_lpdf(muA | 0, 1)
              + normal_lpdf(muN | -1, 0.5)
              + normal_lpdf(tau | 0, 0.3)
              + lkj_corr_cholesky_lpdf(L_Omega | 2)
              + std_normal_lpdf(to_vector(z_IDs));

  // Individual parameters
  array[agents] real<lower=0, upper=1> agent_bias;
  array[agents] real<lower=0, upper=1> agent_noise;

  // Per-trial log-lik (needed for LOO and priorsense)
  array[trials, agents] real log_lik;

  for (i in 1:agents) {
    real alpha_i = muA + IDs[i, 1];
    real noise_i = inv_logit(muN + IDs[i, 2]);

    agent_bias[i]  = inv_logit(alpha_i);
    agent_noise[i] = noise_i;

    for (t in 1:trials)
      log_lik[t, i] = log_mix(noise_i,
                               bernoulli_logit_lpmf(h[t, i] | 0.0),
                               bernoulli_logit_lpmf(h[t, i] | alpha_i));
  }
}

