
// Multilevel Decay Generalized Context Model
// All individual parameters use non-centred parameterisation:
//   - w_1 via logit-normal NCP (eliminates Dirichlet funnel geometry)
//   - log_c, log_lambda, logit_bias via standard z-offset NCP
// Sequential likelihood is computed directly in the model block (no
// prob_cat1 stored in transformed parameters), avoiding a large var-vector
// on the autodiff tape. prob_cat1 and log_lik are recomputed once per saved
// draw in generated quantities.

data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  int<lower=1> N_features;
  int<lower=1> N_unique_stim;
  int<lower=1> max_trials_per_subject;

  array[N_subjects] int<lower=1, upper=N_total> subj_start;
  array[N_subjects] int<lower=1, upper=N_total> subj_end;

  array[N_total] int<lower=1, upper=N_unique_stim> stim_id;
  array[N_total] int<lower=0, upper=1> y;
  array[N_unique_stim, N_features] real unique_obs;
  array[N_total] int<lower=0, upper=1> cat_feedback;

  // Hyperparameters
  real prior_pop_logit_w_mean;
  real<lower=0> prior_pop_logit_w_sigma;
  real<lower=0> prior_pop_logit_w_sd_lambda;
  real prior_pop_log_c_mu;
  real<lower=0> prior_pop_log_c_sigma;
  real<lower=0> prior_pop_log_c_lambda;
  real prior_pop_log_lambda_mu;
  real<lower=0> prior_pop_log_lambda_sigma;
  real<lower=0> prior_pop_log_lambda_lambda;
  real prior_pop_logit_bias_mu;
  real<lower=0> prior_pop_logit_bias_sigma;
  real<lower=0> prior_pop_logit_bias_lambda;
  int<lower=0, upper=1> run_diagnostics;
}

transformed data {
  array[N_unique_stim, N_unique_stim, N_features] real stimulus_abs_diff;
  for (i in 1:N_unique_stim)
    for (k in 1:N_unique_stim)
      for (f in 1:N_features)
        stimulus_abs_diff[i, k, f] = abs(unique_obs[i, f] - unique_obs[k, f]);
}

parameters {
  // Population attention weight: logit-normal NCP for w_1 (w_2 = 1 - w_1)
  real pop_logit_w_mean;
  real<lower=0> pop_logit_w_sd;
  vector[N_subjects] z_w;

  // Population c, lambda, bias
  real pop_log_c_mean;
  real<lower=0> pop_log_c_sd;
  real pop_log_lambda_mean;
  real<lower=0> pop_log_lambda_sd;
  real pop_logit_bias_mean;
  real<lower=0> pop_logit_bias_sd;

  // Non-centred individual offsets
  vector[N_subjects] z_log_c;
  vector[N_subjects] z_log_lambda;
  vector[N_subjects] z_logit_bias;
}

transformed parameters {
  vector[N_subjects] subj_log_c;
  vector[N_subjects] subj_log_lambda;
  vector<lower=0, upper=1>[N_subjects] subj_bias;
  array[N_subjects] vector[N_features] subj_w;

  for (j in 1:N_subjects) {
    subj_log_c[j]      = pop_log_c_mean + pop_log_c_sd * z_log_c[j];
    subj_log_lambda[j] = pop_log_lambda_mean + pop_log_lambda_sd * z_log_lambda[j];
    subj_bias[j]       = inv_logit(pop_logit_bias_mean + pop_logit_bias_sd * z_logit_bias[j]);
    real logit_w       = pop_logit_w_mean + pop_logit_w_sd * z_w[j];
    subj_w[j][1]       = inv_logit(logit_w);
    subj_w[j][2]       = 1.0 - subj_w[j][1];
  }
}

model {
  // Population priors
  pop_logit_w_mean    ~ normal(prior_pop_logit_w_mean, prior_pop_logit_w_sigma);
  pop_logit_w_sd      ~ exponential(prior_pop_logit_w_sd_lambda);
  pop_log_c_mean      ~ normal(prior_pop_log_c_mu, prior_pop_log_c_sigma);
  pop_log_c_sd        ~ exponential(prior_pop_log_c_lambda);
  pop_log_lambda_mean ~ normal(prior_pop_log_lambda_mu, prior_pop_log_lambda_sigma);
  pop_log_lambda_sd   ~ exponential(prior_pop_log_lambda_lambda);
  pop_logit_bias_mean ~ normal(prior_pop_logit_bias_mu, prior_pop_logit_bias_sigma);
  pop_logit_bias_sd   ~ exponential(prior_pop_logit_bias_lambda);

  // Non-centred priors
  z_w          ~ std_normal();
  z_log_c      ~ std_normal();
  z_log_lambda ~ std_normal();
  z_logit_bias ~ std_normal();

  // Sequential likelihood — local workspace, one subject at a time
  {
    array[max_trials_per_subject] int mem_stim;
    array[max_trials_per_subject] int mem_cat;
    array[max_trials_per_subject] int mem_trial;

    for (j in 1:N_subjects) {
      int n_mem  = 0;
      real c      = exp(subj_log_c[j]);
      real lambda = exp(subj_log_lambda[j]);
      real bias   = subj_bias[j];

      for (i in subj_start[j]:subj_end[j]) {
        real p_i;
        if (n_mem == 0) {
          p_i = bias;
        } else {
          real s1 = 0.0; real s0 = 0.0;
          real w1 = 0.0; real w0 = 0.0;
          int curr_stim = stim_id[i];
          for (e in 1:n_mem) {
            real dist = 0.0;
            int past_stim = mem_stim[e];
            for (f in 1:N_features)
              dist += subj_w[j][f] * stimulus_abs_diff[curr_stim, past_stim, f];
            real age   = (i - subj_start[j] + 1.0) - mem_trial[e];
            real decay = exp(-lambda * age);
            real sim   = exp(-c * dist);
            real wt    = decay * sim;
            if (mem_cat[e] == 1) { s1 += wt; w1 += decay; }
            else                 { s0 += wt; w0 += decay; }
          }
          real m1  = (w1 > 1e-9) ? s1 / w1 : 0.0;
          real m0  = (w0 > 1e-9) ? s0 / w0 : 0.0;
          real num = bias * m1;
          real den = num + (1.0 - bias) * m0;
          p_i = (den > 1e-9) ? num / den : bias;
        }
        target += bernoulli_lpmf(y[i] | fmax(1e-9, fmin(1.0 - 1e-9, p_i)));

        n_mem += 1;
        mem_stim[n_mem]  = stim_id[i];
        mem_cat[n_mem]   = cat_feedback[i];
        mem_trial[n_mem] = i - subj_start[j] + 1;
      }
    }
  }
}

generated quantities {
  // Recompute once per saved draw (not on every leapfrog step).
  // prob_cat1 is always filled (needed for PPC); log_lik only when run_diagnostics=1.
  vector[N_total] prob_cat1;
  vector[N_total] log_lik = rep_vector(0.0, N_total);

  {
    array[max_trials_per_subject] int mem_stim;
    array[max_trials_per_subject] int mem_cat;
    array[max_trials_per_subject] int mem_trial;

    for (j in 1:N_subjects) {
      int n_mem  = 0;
      real c      = exp(subj_log_c[j]);
      real lambda = exp(subj_log_lambda[j]);
      real bias   = subj_bias[j];

      for (i in subj_start[j]:subj_end[j]) {
        real p_i;
        if (n_mem == 0) {
          p_i = bias;
        } else {
          real s1 = 0.0; real s0 = 0.0;
          real w1 = 0.0; real w0 = 0.0;
          int curr_stim = stim_id[i];
          for (e in 1:n_mem) {
            real dist = 0.0;
            int past_stim = mem_stim[e];
            for (f in 1:N_features)
              dist += subj_w[j][f] * stimulus_abs_diff[curr_stim, past_stim, f];
            real age   = (i - subj_start[j] + 1.0) - mem_trial[e];
            real decay = exp(-lambda * age);
            real sim   = exp(-c * dist);
            real wt    = decay * sim;
            if (mem_cat[e] == 1) { s1 += wt; w1 += decay; }
            else                 { s0 += wt; w0 += decay; }
          }
          real m1  = (w1 > 1e-9) ? s1 / w1 : 0.0;
          real m0  = (w0 > 1e-9) ? s0 / w0 : 0.0;
          real num = bias * m1;
          real den = num + (1.0 - bias) * m0;
          p_i = (den > 1e-9) ? num / den : bias;
        }
        real p_clamp  = fmax(1e-9, fmin(1.0 - 1e-9, p_i));
        prob_cat1[i]  = p_clamp;
        if (run_diagnostics)
          log_lik[i]  = bernoulli_lpmf(y[i] | p_clamp);

        n_mem += 1;
        mem_stim[n_mem]  = stim_id[i];
        mem_cat[n_mem]   = cat_feedback[i];
        mem_trial[n_mem] = i - subj_start[j] + 1;
      }
    }
  }
}

