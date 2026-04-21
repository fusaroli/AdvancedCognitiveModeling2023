
// Kalman Filter Prototype Model — Single Subject, with Process Noise
// Key design:
//   log_r and log_q are the two parameters (both unconstrained).
//   r_value = exp(log_r): observation noise variance
//   q_value = exp(log_q): process noise / prototype drift rate
//   Kalman loop structure: Predict (add Q) → Decide → Update.
//   The full filter is in transformed parameters so log_lik in generated
//   quantities reads p[i] directly without re-running the filter.

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures;
  array[ntrials] int<lower=0, upper=1> cat_one;
  array[ntrials] int<lower=0, upper=1> y;
  array[ntrials, nfeatures] real obs;

  vector[nfeatures] initial_mu_cat0;
  vector[nfeatures] initial_mu_cat1;
  real<lower=0> initial_sigma_diag;

  real prior_logr_mean;
  real<lower=0> prior_logr_sd;
  real prior_logq_mean;
  real<lower=0> prior_logq_sd;
}

parameters {
  real log_r;
  real log_q;
}

transformed parameters {
  real<lower=0> r_value = exp(log_r);
  real<lower=0> q_value = exp(log_q);

  array[ntrials] real<lower=1e-9, upper=1-1e-9> p;

  {
    vector[nfeatures] mu_cat0 = initial_mu_cat0;
    vector[nfeatures] mu_cat1 = initial_mu_cat1;
    matrix[nfeatures, nfeatures] sigma_cat0 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
    matrix[nfeatures, nfeatures] sigma_cat1 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
    matrix[nfeatures, nfeatures] r_matrix =
      diag_matrix(rep_vector(r_value, nfeatures));
    matrix[nfeatures, nfeatures] q_matrix =
      diag_matrix(rep_vector(q_value, nfeatures));
    matrix[nfeatures, nfeatures] I_mat =
      diag_matrix(rep_vector(1.0, nfeatures));

    for (i in 1:ntrials) {
      vector[nfeatures] x = to_vector(obs[i]);

      // ── Prediction step: add process noise to both categories ──────────
      sigma_cat0 = sigma_cat0 + q_matrix;
      sigma_cat1 = sigma_cat1 + q_matrix;

      // ── Decision ────────────────────────────────────────────────────────
      matrix[nfeatures, nfeatures] cov0 = sigma_cat0 + r_matrix;
      matrix[nfeatures, nfeatures] cov1 = sigma_cat1 + r_matrix;

      real log_p0 = multi_normal_lpdf(x | mu_cat0, cov0);
      real log_p1 = multi_normal_lpdf(x | mu_cat1, cov1);
      real prob1  = exp(log_p1 - log_sum_exp(log_p0, log_p1));

      p[i] = fmax(1e-9, fmin(1 - 1e-9, prob1));

      // ── Update (measurement update for the correct category only) ────────
      if (cat_one[i] == 1) {
        vector[nfeatures] innov = x - mu_cat1;
        matrix[nfeatures, nfeatures] S  = sigma_cat1 + r_matrix;
        matrix[nfeatures, nfeatures] K  = mdivide_right_spd(sigma_cat1, S);
        matrix[nfeatures, nfeatures] IK = I_mat - K;
        mu_cat1    = mu_cat1 + K * innov;
        sigma_cat1 = IK * sigma_cat1 * IK' + K * r_matrix * K';
        sigma_cat1 = 0.5 * (sigma_cat1 + sigma_cat1');
      } else {
        vector[nfeatures] innov = x - mu_cat0;
        matrix[nfeatures, nfeatures] S  = sigma_cat0 + r_matrix;
        matrix[nfeatures, nfeatures] K  = mdivide_right_spd(sigma_cat0, S);
        matrix[nfeatures, nfeatures] IK = I_mat - K;
        mu_cat0    = mu_cat0 + K * innov;
        sigma_cat0 = IK * sigma_cat0 * IK' + K * r_matrix * K';
        sigma_cat0 = 0.5 * (sigma_cat0 + sigma_cat0');
      }
    }
  }
}

model {
  target += normal_lpdf(log_r | prior_logr_mean, prior_logr_sd);
  target += normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);
  target += bernoulli_lpmf(y | p);
}

generated quantities {
  vector[ntrials] log_lik;
  for (i in 1:ntrials)
    log_lik[i] = bernoulli_lpmf(y[i] | p[i]);
}

