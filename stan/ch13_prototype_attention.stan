
// Prototype Model with Selective Attention — Feature-Space Rescaling
//
// Parameters:
//   log_r  : log observation noise (same role as in the base model)
//   log_q  : log process noise     (same role as in the base model)
//   w      : attention simplex (length = nfeatures, sums to 1)
//
// Attention mechanism:
//   Before any computation on trial i, we replace the raw stimulus x[i]
//   with its attention-rescaled version  x_tilde = w .* x[i]  (element-wise).
//   Initial prototype means are also rescaled: mu_init_tilde = w .* mu_init.
//   The Kalman filter then runs entirely in this rescaled space.
//   w and r are NOT redundant: w controls the *shape* of the decision
//   boundary (which features matter); r controls overall decisional
//   sharpness. Redundancy would arise if we had instead written
//   R = diag(r ./ w), which Option 1 would require.

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

  // Dirichlet concentration parameters for the attention prior.
  // alpha = rep_vector(1, nfeatures) gives a uniform prior over the simplex:
  // every allocation of attention is equally likely a priori.
  vector<lower=0>[nfeatures] alpha_w;
}

parameters {
  real log_r;
  real log_q;
  // Stan's simplex type enforces w_k >= 0 and sum(w) = 1 automatically.
  simplex[nfeatures] w;
}

transformed parameters {
  real<lower=0> r_value = exp(log_r);
  real<lower=0> q_value = exp(log_q);

  array[ntrials] real<lower=1e-9, upper=1-1e-9> p;

  {
    // Rescale the initial means into the attention-weighted space.
    // This keeps the initial prototype location consistent with the
    // rescaled observations that the filter will receive.
    vector[nfeatures] mu_cat0 = w .* initial_mu_cat0;
    vector[nfeatures] mu_cat1 = w .* initial_mu_cat1;

    matrix[nfeatures, nfeatures] sigma_cat0 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
    matrix[nfeatures, nfeatures] sigma_cat1 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));

    // r and q matrices are unchanged: attention acts on the data, not
    // on the noise structure (this is the key difference from Option 1).
    matrix[nfeatures, nfeatures] r_matrix =
      diag_matrix(rep_vector(r_value, nfeatures));
    matrix[nfeatures, nfeatures] q_matrix =
      diag_matrix(rep_vector(q_value, nfeatures));
    matrix[nfeatures, nfeatures] I_mat =
      diag_matrix(rep_vector(1.0, nfeatures));

    for (i in 1:ntrials) {
      // Rescale the raw stimulus by attention weights before any computation.
      // This is the only line that differs from the base model's loop.
      vector[nfeatures] x = w .* to_vector(obs[i]);

      // ── Prediction step (unchanged from base model) ────────────────────────
      sigma_cat0 = sigma_cat0 + q_matrix;
      sigma_cat1 = sigma_cat1 + q_matrix;

      // ── Decision (unchanged from base model) ──────────────────────────────
      // x is now in the rescaled space; prototypes are also in the rescaled
      // space. The Mahalanobis distance is computed in that common space.
      matrix[nfeatures, nfeatures] cov0 = sigma_cat0 + r_matrix;
      matrix[nfeatures, nfeatures] cov1 = sigma_cat1 + r_matrix;

      real log_p0 = multi_normal_lpdf(x | mu_cat0, cov0);
      real log_p1 = multi_normal_lpdf(x | mu_cat1, cov1);
      real prob1  = exp(log_p1 - log_sum_exp(log_p0, log_p1));

      p[i] = fmax(1e-9, fmin(1 - 1e-9, prob1));

      // ── Update (unchanged from base model) ────────────────────────────────
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
  // Priors on noise parameters: same as base model.
  target += normal_lpdf(log_r | prior_logr_mean, prior_logr_sd);
  target += normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);

  // Prior on attention weights: uniform over the simplex.
  // Dirichlet(1, 1, ...) = no preference for any feature a priori.
  // To favour equal attention (e.g., a regularisation prior), increase alpha.
  target += dirichlet_lpdf(w | alpha_w);

  target += bernoulli_lpmf(y | p);
}

generated quantities {
  vector[ntrials] log_lik;
  real lprior;
  for (i in 1:ntrials)
    log_lik[i] = bernoulli_lpmf(y[i] | p[i]);
  lprior = normal_lpdf(log_r | prior_logr_mean, prior_logr_sd) +
           normal_lpdf(log_q | prior_logq_mean, prior_logq_sd) +
           dirichlet_lpdf(w | alpha_w);
}

