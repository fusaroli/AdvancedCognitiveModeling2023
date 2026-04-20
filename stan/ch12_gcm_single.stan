
// Generalized Context Model — Single Subject (refactored architecture)
//
// Key design:
//   prob_cat1[i] is the per-trial choice probability. It is deterministic
//   given (w, log_c, bias) and the data sequence, so it lives in
//   transformed parameters. model{} reads it once via bernoulli_lpmf.
//   generated quantities{} reads the same vector for log_lik[i].
//
// Parameterization:
//   w      : attention weights (simplex)
//   log_c  : log sensitivity (unconstrained); c = exp(log_c)
//   bias   : response bias towards category 1 (0–1)

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures;
  array[ntrials] int<lower=0, upper=1> y;
  array[ntrials, nfeatures] real obs;
  array[ntrials] int<lower=0, upper=1> cat_feedback;

  vector[nfeatures] w_prior_alpha;
  real log_c_prior_mean;
  real<lower=0> log_c_prior_sd;
  real<lower=0> bias_prior_alpha;
  real<lower=0> bias_prior_beta;
}

transformed data {
  array[ntrials, ntrials, nfeatures] real abs_diff;

  for (i in 1:ntrials) {
    for (j in 1:ntrials) {
      for (f in 1:nfeatures) {
        abs_diff[i, j, f] = abs(obs[i, f] - obs[j, f]);
      }
    }
  }
}

parameters {
  simplex[nfeatures] w;
  real log_c;
  real<lower=0, upper=1> bias;
}

transformed parameters {
  real<lower=0> c = exp(log_c);

  vector<lower=1e-9, upper=1-1e-9>[ntrials] prob_cat1;

  {
    array[ntrials] int memory_trial_idx;
    array[ntrials] int memory_cat;
    int n_mem = 0;

    for (i in 1:ntrials) {
      real p_i;
      int has_cat0 = 0;
      int has_cat1 = 0;

      for (k in 1:n_mem) {
        if (memory_cat[k] == 0) has_cat0 = 1;
        if (memory_cat[k] == 1) has_cat1 = 1;
      }

      if (n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) {
        p_i = bias;
      } else {
        vector[n_mem] sims;
        for (e in 1:n_mem) {
          real d = 0;
          int past_i = memory_trial_idx[e];
          for (f in 1:nfeatures)
            d += w[f] * abs_diff[i, past_i, f];
          sims[e] = exp(-c * d);
        }
        real s1 = 0;
        real s0 = 0;
        for (e in 1:n_mem) {
          if (memory_cat[e] == 1) s1 += sims[e];
          else                    s0 += sims[e];
        }
        real num = bias * s1;
        real den = num + (1 - bias) * s0;
        p_i = (den > 1e-9) ? num / den : bias;
      }

      prob_cat1[i] = fmax(1e-9, fmin(1 - 1e-9, p_i));

      n_mem += 1;
      memory_trial_idx[n_mem] = i;
      memory_cat[n_mem] = cat_feedback[i];
    }
  }
}

model {
  target += dirichlet_lpdf(w    | w_prior_alpha);
  target += normal_lpdf(log_c   | log_c_prior_mean, log_c_prior_sd);
  target += beta_lpdf(bias      | bias_prior_alpha, bias_prior_beta);

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[ntrials] log_lik;
  real lprior;
  for (i in 1:ntrials)
    log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
  lprior = dirichlet_lpdf(w | w_prior_alpha) +
           normal_lpdf(log_c | log_c_prior_mean, log_c_prior_sd) +
           beta_lpdf(bias | bias_prior_alpha, bias_prior_beta);
}

