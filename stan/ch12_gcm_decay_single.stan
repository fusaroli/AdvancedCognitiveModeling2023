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
// which at lambda = 0 becomes the plain mean similarity used in
// ch12_gcm_single.stan. The denominator sum of decay weights serves as
// an effective sample size and removes the category-frequency bias of
// the canonical summed form.
//
// Parameterisation:
//   w          : attention weights (simplex)
//   log_c      : log sensitivity; c = exp(log_c)
//   log_lambda : log forgetting rate; lambda = exp(log_lambda)
//   bias       : response bias toward category 1 (0–1)
//
// The per-trial choice probability prob_cat1[i] is fully deterministic
// given (w, c, lambda, bias) and the data sequence, so it lives in
// transformed parameters. model{} reads it via one vectorised
// bernoulli_lpmf; generated quantities{} reads the same vector for
// log_lik[i].

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures;
  array[ntrials] int<lower=0, upper=1> y;
  array[ntrials, nfeatures] real obs;
  array[ntrials] int<lower=0, upper=1> cat_feedback;

  vector[nfeatures] w_prior_alpha;
  real log_c_prior_mean;
  real<lower=0> log_c_prior_sd;
  real log_lambda_prior_mean;
  real<lower=0> log_lambda_prior_sd;
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
  real log_lambda;
  real<lower=0, upper=1> bias;
}

transformed parameters {
  real<lower=0> c      = exp(log_c);
  real<lower=0> lambda = exp(log_lambda);

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
        real s0 = 0;   // sum of decay * similarity for cat 0
        real s1 = 0;   // sum of decay * similarity for cat 1
        real w0 = 0;   // sum of decay weights for cat 0 (effective N)
        real w1 = 0;   // sum of decay weights for cat 1 (effective N)
        for (e in 1:n_mem) {
          real d = 0;
          int past_i = memory_trial_idx[e];
          for (f in 1:nfeatures)
            d += w[f] * abs_diff[i, past_i, f];
          real age          = i - past_i;       // age in trials (>= 1)
          real decay_weight = exp(-lambda * age);
          real sim          = exp(-c * d);
          real contrib      = decay_weight * sim;
          if (memory_cat[e] == 1) { s1 += contrib; w1 += decay_weight; }
          else                    { s0 += contrib; w0 += decay_weight; }
        }
        // Decay-weighted mean similarity per category. At lambda = 0,
        // w1 / w0 collapse to category counts and m1, m0 are plain means.
        real m1 = (w1 > 1e-12) ? s1 / w1 : 0;
        real m0 = (w0 > 1e-12) ? s0 / w0 : 0;
        real num = bias * m1;
        real den = num + (1 - bias) * m0;
        p_i = (den > 1e-12) ? num / den : bias;
      }

      prob_cat1[i] = fmax(1e-9, fmin(1 - 1e-9, p_i));

      n_mem += 1;
      memory_trial_idx[n_mem] = i;
      memory_cat[n_mem] = cat_feedback[i];
    }
  }
}

model {
  target += dirichlet_lpdf(w | w_prior_alpha);
  target += normal_lpdf(log_c      | log_c_prior_mean,      log_c_prior_sd);
  target += normal_lpdf(log_lambda | log_lambda_prior_mean, log_lambda_prior_sd);
  target += beta_lpdf(bias         | bias_prior_alpha,      bias_prior_beta);

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[ntrials] log_lik;
  real lprior;
  for (i in 1:ntrials)
    log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
  lprior =
    dirichlet_lpdf(w         | w_prior_alpha) +
    normal_lpdf(log_c        | log_c_prior_mean,      log_c_prior_sd) +
    normal_lpdf(log_lambda   | log_lambda_prior_mean, log_lambda_prior_sd) +
    beta_lpdf(bias           | bias_prior_alpha,      bias_prior_beta);
}
