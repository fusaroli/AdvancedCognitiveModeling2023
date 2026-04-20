
// Generalized Context Model — Multilevel (refactored architecture)
//
// New relative to the single-subject model:
//   1. log_c parameterization with logit-normal hierarchy for bias.
//   2. Subject-indexed loop: outer over subjects, inner over each subject's
//      contiguous slice [subj_start[j]:subj_end[j]].
//   3. prob_cat1[i] in transformed parameters; model{} and generated quantities
//      both read from it without re-running the loop.
//   4. ALL prior hyperparameters passed through the data block.

data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  int<lower=1> N_features;
  int<lower=1> max_trials_per_subject;

  array[N_subjects] int<lower=1, upper=N_total> subj_start;
  array[N_subjects] int<lower=1, upper=N_total> subj_end;

  array[N_total] int<lower=0, upper=1> y;
  array[N_total, N_features] real obs;
  array[N_total] int<lower=0, upper=1> cat_feedback;

  vector[N_features] pop_w_prior_alpha;
  real pop_log_c_mean_prior_mean;
  real<lower=0> pop_log_c_mean_prior_sd;
  real<lower=0> pop_log_c_sd_prior_rate;
  real<lower=0> kappa_prior_rate;
  real pop_logit_bias_mean_prior_mean;
  real<lower=0> pop_logit_bias_mean_prior_sd;
  real<lower=0> pop_logit_bias_sd_prior_rate;
}

parameters {
  simplex[N_features] pop_w;
  real<lower=0> kappa;
  real pop_log_c_mean;
  real<lower=0> pop_log_c_sd;
  real pop_logit_bias_mean;
  real<lower=0> pop_logit_bias_sd;

  vector[N_subjects] z_log_c;
  vector[N_subjects] z_logit_bias;

  array[N_subjects] simplex[N_features] subj_w;
}

transformed parameters {
  vector[N_subjects] subj_log_c;
  vector<lower=0>[N_subjects] subj_c;
  vector<lower=0, upper=1>[N_subjects] subj_bias;

  vector<lower=1e-9, upper=1-1e-9>[N_total] prob_cat1;

  for (j in 1:N_subjects) {
    subj_log_c[j] = pop_log_c_mean + z_log_c[j] * pop_log_c_sd;
    subj_c[j]     = exp(subj_log_c[j]);
    subj_bias[j]  = inv_logit(pop_logit_bias_mean + z_logit_bias[j] * pop_logit_bias_sd);
  }

  {
    array[N_subjects, max_trials_per_subject, N_features] real memory_obs;
    array[N_subjects, max_trials_per_subject] int memory_cat;
    array[N_subjects] int n_mem_subj;

    for (j in 1:N_subjects) n_mem_subj[j] = 0;

    for (j in 1:N_subjects) {
      int n_mem = 0;

      for (i in subj_start[j]:subj_end[j]) {
        real p_i;
        int has_cat0 = 0;
        int has_cat1 = 0;

        for (k in 1:n_mem) {
          if (memory_cat[j, k] == 0) has_cat0 = 1;
          if (memory_cat[j, k] == 1) has_cat1 = 1;
        }

        if (n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) {
          p_i = subj_bias[j];
        } else {
          vector[n_mem] sims;
          for (e in 1:n_mem) {
            real d = 0;
            for (f in 1:N_features)
              d += subj_w[j][f] * abs(obs[i, f] - memory_obs[j, e, f]);
            sims[e] = exp(-subj_c[j] * d);
          }
          real s1 = 0;
          real s0 = 0;
          for (e in 1:n_mem) {
            if (memory_cat[j, e] == 1) s1 += sims[e];
            else                       s0 += sims[e];
          }
          real num = subj_bias[j] * s1;
          real den = num + (1 - subj_bias[j]) * s0;
          p_i = (den > 1e-9) ? num / den : subj_bias[j];
        }

        prob_cat1[i] = fmax(1e-9, fmin(1 - 1e-9, p_i));

        n_mem += 1;
        for (f in 1:N_features) memory_obs[j, n_mem, f] = obs[i, f];
        memory_cat[j, n_mem] = cat_feedback[i];
      }
    }
  }
}

model {
  target += dirichlet_lpdf(pop_w           | pop_w_prior_alpha);
  target += exponential_lpdf(kappa         | kappa_prior_rate);
  target += normal_lpdf(pop_log_c_mean     | pop_log_c_mean_prior_mean,
                                              pop_log_c_mean_prior_sd);
  target += exponential_lpdf(pop_log_c_sd  | pop_log_c_sd_prior_rate);
  target += normal_lpdf(pop_logit_bias_mean | pop_logit_bias_mean_prior_mean,
                                              pop_logit_bias_mean_prior_sd);
  target += exponential_lpdf(pop_logit_bias_sd | pop_logit_bias_sd_prior_rate);

  target += std_normal_lpdf(z_log_c);
  target += std_normal_lpdf(z_logit_bias);

  for (j in 1:N_subjects) {
    vector[N_features] alpha = kappa * pop_w;
    for (f in 1:N_features) alpha[f] = fmax(1e-9, alpha[f]);
    target += dirichlet_lpdf(subj_w[j] | alpha);
  }

  target += bernoulli_lpmf(y | prob_cat1);
}

generated quantities {
  vector[N_total] log_lik;
  for (i in 1:N_total)
    log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1[i]);
}

