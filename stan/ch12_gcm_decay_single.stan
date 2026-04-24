// Generalized Context Model with Memory Decay — Single Subject
//
// Extension of ch12_gcm_single.stan adding a recency-weight parameter
// log_lambda. Each stored exemplar is downweighted by exp(-lambda * age)
// in the category-similarity average, where age = i - trial_of_exemplar.
// The classical GCM is recovered in the limit lambda -> 0.
//
// Category evidence is a decay-weighted *mean* similarity:
//   s_A = sum_{j in A} exp(-lambda * age_j) * eta(i, j)
//         / sum_{j in A} exp(-lambda * age_j)

data {
  int<lower=1> N_total;
  int<lower=1> N_features;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total, N_features] real obs;
  array[N_total] int<lower=0, upper=1> cat_feedback;

  vector[N_features] prior_w_alpha;
  real prior_log_c_mu;
  real<lower=0> prior_log_c_sigma;
  real prior_log_lambda_mu;
  real<lower=0> prior_log_lambda_sigma;
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
  real log_lambda;
  real<lower=0, upper=1> bias;
}

transformed parameters {
  real<lower=0> c      = exp(log_c);
  real<lower=0> lambda = exp(log_lambda);

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
        real s0 = 0.0;   // sum of decay * similarity for cat 0
        real s1 = 0.0;   // sum of decay * similarity for cat 1
        real w0 = 0.0;   // sum of decay weights for cat 0 (effective N)
        real w1 = 0.0;   // sum of decay weights for cat 1 (effective N)
        for (e in 1:n_mem) {
          real d = 0.0;
          int past_i = memory_trial_idx[e];
          for (f in 1:N_features)
            d += w[f] * abs_diff[i, past_i, f];
          real age          = i - past_i;
          real decay_weight = exp(-lambda * age);
          real sim          = exp(-c * d);
          real weight       = decay_weight * sim;
          if (memory_cat[e] == 1) { s1 += weight; w1 += decay_weight; }
          else                    { s0 += weight; w0 += decay_weight; }
        }
        real m1 = (w1 > 1e-12) ? s1 / w1 : 0.0;
        real m0 = (w0 > 1e-12) ? s0 / w0 : 0.0;
        real num = bias * m1;
        real den = num + (1.0 - bias) * m0;
        p_i = (den > 1e-12) ? num / den : bias;
      }

      prob_cat1[i] = fmax(1e-9, fmin(1.0 - 1e-9, p_i));

      n_mem += 1;
      memory_trial_idx[n_mem] = i;
      memory_cat[n_mem] = cat_feedback[i];
    }
  }
}

model {
  target += dirichlet_lpdf(w | prior_w_alpha);
  target += normal_lpdf(log_c      | prior_log_c_mu,      prior_log_c_sigma);
  target += normal_lpdf(log_lambda | prior_log_lambda_mu, prior_log_lambda_sigma);
  target += beta_lpdf(bias         | prior_bias_alpha,    prior_bias_beta);

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[N_total] log_lik;
  real lprior;
  
  if (run_diagnostics) {
    for (i in 1:N_total)
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
  }
  
  lprior =
    dirichlet_lpdf(w         | prior_w_alpha) +
    normal_lpdf(log_c        | prior_log_c_mu,      prior_log_c_sigma) +
    normal_lpdf(log_lambda   | prior_log_lambda_mu, prior_log_lambda_sigma) +
    beta_lpdf(bias           | prior_bias_alpha,    prior_bias_beta);
}
