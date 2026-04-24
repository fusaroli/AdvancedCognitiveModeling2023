
// Kalman Filter Prototype Model — Multilevel (Non-Centred Parameterisation)
//
// New relative to the single-subject model:
//   1. Population hyperparameters for log_r AND log_q.
//   2. Per-subject NCP offsets z_log_r[j] and z_log_q[j] ~ Normal(0, 1).
//   3. Subject-indexed Kalman filter loop with prediction step in transformed parameters.
//   4. ALL prior hyperparameters passed through the data block.

data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  int<lower=1> N_features;

  array[N_subjects] int<lower=1, upper=N_total> subj_start;
  array[N_subjects] int<lower=1, upper=N_total> subj_end;

  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0, upper=1> cat_one;
  array[N_total, N_features] real obs;

  // Fixed structural values (same as single-subject; not inferred)
  vector[N_features] initial_mu_cat0;
  vector[N_features] initial_mu_cat1;
  real<lower=0> initial_sigma_diag;

  // Prior hyperparameters for r
  real pop_log_r_mean_prior_mean;
  real<lower=0> pop_log_r_mean_prior_sd;
  real<lower=0> pop_log_r_sd_prior_rate;

  // Prior hyperparameters for q
  real pop_log_q_mean_prior_mean;
  real<lower=0> pop_log_q_mean_prior_sd;
  real<lower=0> pop_log_q_sd_prior_rate;
}

parameters {
  real pop_log_r_mean;
  real<lower=0> pop_log_r_sd;
  vector[N_subjects] z_log_r;

  real pop_log_q_mean;
  real<lower=0> pop_log_q_sd;
  vector[N_subjects] z_log_q;
}

transformed parameters {
  vector[N_subjects] subj_log_r;
  vector<lower=0>[N_subjects] subj_r;
  vector[N_subjects] subj_log_q;
  vector<lower=0>[N_subjects] subj_q;
  vector<lower=1e-9, upper=1-1e-9>[N_total] prob_cat1;

  for (j in 1:N_subjects) {
    subj_log_r[j] = pop_log_r_mean + z_log_r[j] * pop_log_r_sd;
    subj_r[j]     = exp(subj_log_r[j]);
    subj_log_q[j] = pop_log_q_mean + z_log_q[j] * pop_log_q_sd;
    subj_q[j]     = exp(subj_log_q[j]);
  }

  {
    matrix[N_features, N_features] I_mat =
      diag_matrix(rep_vector(1.0, N_features));

    for (j in 1:N_subjects) {
      vector[N_features] mu_cat0 = initial_mu_cat0;
      vector[N_features] mu_cat1 = initial_mu_cat1;
      matrix[N_features, N_features] sigma_cat0 =
        diag_matrix(rep_vector(initial_sigma_diag, N_features));
      matrix[N_features, N_features] sigma_cat1 =
        diag_matrix(rep_vector(initial_sigma_diag, N_features));
      matrix[N_features, N_features] r_matrix =
        diag_matrix(rep_vector(subj_r[j], N_features));
      matrix[N_features, N_features] q_matrix =
        diag_matrix(rep_vector(subj_q[j], N_features));

      for (i in subj_start[j]:subj_end[j]) {
        vector[N_features] x = to_vector(obs[i]);

        // ── Prediction step: add process noise to both categories ──────────
        sigma_cat0 = sigma_cat0 + q_matrix;
        sigma_cat1 = sigma_cat1 + q_matrix;

        // ── Decision ────────────────────────────────────────────────────────
        matrix[N_features, N_features] cov0 = sigma_cat0 + r_matrix;
        matrix[N_features, N_features] cov1 = sigma_cat1 + r_matrix;
        real log_p0 = multi_normal_lpdf(x | mu_cat0, cov0);
        real log_p1 = multi_normal_lpdf(x | mu_cat1, cov1);
        real p_i    = exp(log_p1 - log_sum_exp(log_p0, log_p1));
        prob_cat1[i] = fmax(1e-9, fmin(1 - 1e-9, p_i));

        // ── Update ───────────────────────────────────────────────────────────
        if (cat_one[i] == 1) {
          vector[N_features] innov = x - mu_cat1;
          matrix[N_features, N_features] S  = sigma_cat1 + r_matrix;
          matrix[N_features, N_features] K  = mdivide_right_spd(sigma_cat1, S);
          matrix[N_features, N_features] IK = I_mat - K;
          mu_cat1    = mu_cat1 + K * innov;
          sigma_cat1 = IK * sigma_cat1 * IK' + K * r_matrix * K';
          sigma_cat1 = 0.5 * (sigma_cat1 + sigma_cat1');
        } else {
          vector[N_features] innov = x - mu_cat0;
          matrix[N_features, N_features] S  = sigma_cat0 + r_matrix;
          matrix[N_features, N_features] K  = mdivide_right_spd(sigma_cat0, S);
          matrix[N_features, N_features] IK = I_mat - K;
          mu_cat0    = mu_cat0 + K * innov;
          sigma_cat0 = IK * sigma_cat0 * IK' + K * r_matrix * K';
          sigma_cat0 = 0.5 * (sigma_cat0 + sigma_cat0');
        }
      }
    }
  }
}

model {
  target += normal_lpdf(pop_log_r_mean | pop_log_r_mean_prior_mean,
                                         pop_log_r_mean_prior_sd);
  target += exponential_lpdf(pop_log_r_sd | pop_log_r_sd_prior_rate);
  target += std_normal_lpdf(z_log_r);

  target += normal_lpdf(pop_log_q_mean | pop_log_q_mean_prior_mean,
                                         pop_log_q_mean_prior_sd);
  target += exponential_lpdf(pop_log_q_sd | pop_log_q_sd_prior_rate);
  target += std_normal_lpdf(z_log_q);

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[N_total] log_lik;
  real lprior;
  for (i in 1:N_total)
    log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
  lprior = normal_lpdf(pop_log_r_mean | pop_log_r_mean_prior_mean, pop_log_r_mean_prior_sd) +
           exponential_lpdf(pop_log_r_sd | pop_log_r_sd_prior_rate) +
           std_normal_lpdf(z_log_r) +
           normal_lpdf(pop_log_q_mean | pop_log_q_mean_prior_mean, pop_log_q_mean_prior_sd) +
           exponential_lpdf(pop_log_q_sd | pop_log_q_sd_prior_rate) +
           std_normal_lpdf(z_log_q);
}

