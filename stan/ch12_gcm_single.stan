
// Generalized Context Model — Single Subject (refactored architecture)

data {
  int<lower=1> N_total;
  int<lower=1> N_features;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total, N_features] real obs;
  array[N_total] int<lower=0, upper=1> cat_feedback;

  vector[N_features] prior_w_alpha;
  real prior_log_c_mu;
  real<lower=0> prior_log_c_sigma;
  real<lower=0> prior_bias_alpha;
  real<lower=0> prior_bias_beta;
  int<lower=0, upper=1> run_diagnostics;
}

transformed data {
  array[N_total, N_total, N_features] real abs_diff;
  for (i in 1:N_total) {
    for (j in 1:N_total) {
      for (f in 1:N_features) {
        abs_diff[i, j, f] = abs(obs[i, f] - obs[j, f]);
      }
    }
  }
}

parameters {
  simplex[N_features] w;
  real log_c;
  real<lower=0, upper=1> bias;
}

transformed parameters {
  real<lower=0> c = exp(log_c);
  vector<lower=1e-9, upper=1-1e-9>[N_total] prob_cat1;

  {
    array[N_total] int memory_trial_idx;
    array[N_total] int memory_cat;
    int n_mem = 0;

    for (i in 1:N_total) {
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
        real s1 = 0; real s0 = 0;
        int n1 = 0;  int n0 = 0;
        for (e in 1:n_mem) {
          real d = 0;
          int past_i = memory_trial_idx[e];
          for (f in 1:N_features)
            d += w[f] * abs_diff[i, past_i, f];
          real sim = exp(-c * d);
          if (memory_cat[e] == 1) { s1 += sim; n1 += 1; }
          else                    { s0 += sim; n0 += 1; }
        }
        real m1 = s1 / n1;
        real m0 = s0 / n0;
        real num = bias * m1;
        real den = num + (1 - bias) * m0;
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
  target += dirichlet_lpdf(w    | prior_w_alpha);
  target += normal_lpdf(log_c   | prior_log_c_mu, prior_log_c_sigma);
  target += beta_lpdf(bias      | prior_bias_alpha,  prior_bias_beta);

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[N_total] log_lik;
  real lprior;
  if (run_diagnostics) {
    for (i in 1:N_total)
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
  }
  lprior = dirichlet_lpdf(w | prior_w_alpha) +
           normal_lpdf(log_c | prior_log_c_mu, prior_log_c_sigma) +
           beta_lpdf(bias | prior_bias_alpha, prior_bias_beta);
}

