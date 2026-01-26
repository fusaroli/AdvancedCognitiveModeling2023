
// Generalized Context Model (GCM) - Multilevel (Scaled Logit c)

data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  int<lower=1> N_trials;      // Max trials per subject for memory sizing
  int<lower=1> N_features;
  array[N_total] int<lower=1, upper=N_subjects> subj_id;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total, N_features] real obs;
  array[N_total] int<lower=0, upper=1> cat_feedback;

  // Priors - Population level
  vector[N_features] pop_w_prior_alpha;
  array[2] real pop_c_logit_scaled_mean_prior_params; // Prior for pop mean of c_logit_scaled
  array[2] real pop_c_logit_scaled_sd_prior_params;   // Prior for pop sd of c_logit_scaled
  int<lower=0,upper=1> pop_c_logit_scaled_sd_prior_type; // 0=Normal T[0,], 1=Exponential
  array[2] real pop_bias_prior_params; // Prior for population bias mean (e.g., beta)
  real kappa_prior_rate;           // Prior for kappa (e.g., exponential)
  real bias_phi_prior_rate;        // Prior for bias concentration phi (e.g., exponential)

  // Bounds for c
  real<lower=0> C_UPPER_BOUND;
  real LOWER_BOUND; // Typically 0
}

parameters {
  // Population-level parameters
  simplex[N_features] pop_w;
  real<lower=0> kappa;
  real pop_c_logit_scaled_mean;      // Population mean of scaled logit(c)
  real<lower=0> pop_c_logit_scaled_sd; // Population sd of scaled logit(c)
  real<lower=0, upper=1> pop_bias_mean;
  real<lower=0> pop_bias_phi;

  // Individual-level parameters (non-centered for c and bias)
  vector[N_subjects] z_c_logit_scaled; // Standardized deviations for c_logit_scaled
  vector[N_subjects] z_bias;           // Standardized deviations for bias (logit scale)
  array[N_subjects] simplex[N_features] subj_w; // Individual weights (centered easier here)
}

transformed parameters {
  // Transform individual parameters back to scale used in likelihood/simulation
  vector<lower=LOWER_BOUND, upper=C_UPPER_BOUND>[N_subjects] subj_c; // Individual sensitivity (original scale)
  vector<lower=0, upper=1>[N_subjects] subj_bias; // Individual bias (original scale)
  vector[N_subjects] subj_c_logit_scaled; // Individual sensitivity (transformed scale) - for GQ

  for (j in 1:N_subjects) {
    // Calculate individual c_logit_scaled using non-centered parameterization
    subj_c_logit_scaled[j] = pop_c_logit_scaled_mean + z_c_logit_scaled[j] * pop_c_logit_scaled_sd;

    // Derive individual c on original scale
    subj_c[j] = LOWER_BOUND + (C_UPPER_BOUND - LOWER_BOUND) * inv_logit(subj_c_logit_scaled[j]);

    // Derive individual bias using non-centered parameterization (logit scale)
    // Ensure pop_bias_phi is reasonably constrained or use a robust sd calculation
    real bias_sd_approx = sqrt(1.0 / (pop_bias_phi + 1e-9)); // Approximation, avoid division by zero
    subj_bias[j] = inv_logit(logit(pop_bias_mean) + z_bias[j] * bias_sd_approx);
  }
}

model {
  // Priors for population-level parameters
  target += dirichlet_lpdf(pop_w | pop_w_prior_alpha);
  target += exponential_lpdf(kappa | kappa_prior_rate);

  // Priors for TRANSFORMED sensitivity population parameters
  target += normal_lpdf(pop_c_logit_scaled_mean | pop_c_logit_scaled_mean_prior_params[1], pop_c_logit_scaled_mean_prior_params[2]);
  if (pop_c_logit_scaled_sd_prior_type == 0) { // Normal T[0,]
      target += normal_lpdf(pop_c_logit_scaled_sd | pop_c_logit_scaled_sd_prior_params[1], pop_c_logit_scaled_sd_prior_params[2])
                - normal_lccdf(0 | pop_c_logit_scaled_sd_prior_params[1], pop_c_logit_scaled_sd_prior_params[2]);
  } else { // Exponential
      target += exponential_lpdf(pop_c_logit_scaled_sd | pop_c_logit_scaled_sd_prior_params[1]);
  }

  target += beta_lpdf(pop_bias_mean | pop_bias_prior_params[1], pop_bias_prior_params[2]);
  target += exponential_lpdf(pop_bias_phi | bias_phi_prior_rate);

  // Priors for individual deviations (non-centered)
  target += std_normal_lpdf(z_c_logit_scaled);
  target += std_normal_lpdf(z_bias);

  // Hierarchical prior for individual weights (centered)
  for (j in 1:N_subjects) {
    // Ensure alpha parameters are positive for Dirichlet
    vector[N_features] dirichlet_alpha = kappa * pop_w;
    for (f in 1:N_features) {
        dirichlet_alpha[f] = fmax(1e-9, dirichlet_alpha[f]);
    }
    target += dirichlet_lpdf(subj_w[j] | dirichlet_alpha);
  }

  // Likelihood calculation
  {
    // Memory storage needs to be per subject
    // Sizing N_trials assumes max possible trials, might be inefficient if trials vary greatly
    array[N_subjects, N_trials, N_features] real memory_obs;
    array[N_subjects, N_trials] int memory_cat;
    array[N_subjects] int n_memory; // Counter per subject
    // Initialize memory counters outside loop
    for (j in 1:N_subjects) { n_memory[j] = 0; }

    for (i in 1:N_total) {
      int s = subj_id[i];
      real prob_cat1;
      // Use subject-specific parameters derived in transformed parameters block
      vector[N_features] current_w = subj_w[s];
      real current_c = subj_c[s];       // Use derived original c
      real current_bias = subj_bias[s]; // Use derived bias
      int current_n_mem = n_memory[s];
      int has_cat0 = 0; int has_cat1 = 0;

      // Check memory state
      if (current_n_mem > 0) {
         for (k in 1:current_n_mem) {
           if (memory_cat[s, k] == 0) has_cat0 = 1;
           if (memory_cat[s, k] == 1) has_cat1 = 1;
         }
      }

      // Calculate choice probability
      if (current_n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) {
        prob_cat1 = current_bias;
      } else {
        vector[current_n_mem] current_similarities;
        for (e in 1:current_n_mem) {
          real dist_val = 0;
          for (f in 1:N_features) {
             dist_val += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]);
          }
          // Similarity calculation uses the derived original current_c
          current_similarities[e] = exp(-current_c * dist_val);
        }
        real sum_sim_cat1 = 0; real sum_sim_cat0 = 0;
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) sum_sim_cat1 += current_similarities[e];
          else sum_sim_cat0 += current_similarities[e];
        }
        real numerator = current_bias * sum_sim_cat1;
        real denominator = numerator + (1 - current_bias) * sum_sim_cat0;
        if (denominator > 1e-9) prob_cat1 = numerator / denominator;
        else prob_cat1 = current_bias;
        prob_cat1 = fmax(1e-9, fmin(1.0 - 1e-9, prob_cat1));
      }

      target += bernoulli_lpmf(y[i] | prob_cat1); // Likelihood contribution

      // Update memory for subject s
      int current_trial_idx = n_memory[s] + 1;
      // Check if memory update is within bounds
      if (current_trial_idx <= N_trials) {
          for(f in 1:N_features) memory_obs[s, current_trial_idx, f] = obs[i, f];
          memory_cat[s, current_trial_idx] = cat_feedback[i];
          n_memory[s] = current_trial_idx; // Increment memory counter for subject s
      } else {
         // Optional: print warning or handle memory overflow if N_trials is too small
         // print("Warning: Memory overflow for subject ", s);
      }
    }
  }
}

generated quantities {
  vector[N_total] log_lik;
  // Replicate parameters for output, using vector type to match source variables
  vector[N_subjects] subj_c_rep = subj_c; // Original scale c
  vector[N_subjects] subj_c_logit_scaled_rep = subj_c_logit_scaled; // Transformed scale c
  vector[N_subjects] subj_bias_rep = subj_bias; // Replicated bias

  { // Replicate memory logic exactly from model block
    array[N_subjects, N_trials, N_features] real memory_obs;
    array[N_subjects, N_trials] int memory_cat;
    array[N_subjects] int n_memory;
    for (j in 1:N_subjects) { n_memory[j] = 0; }

    for (i in 1:N_total) {
      int s = subj_id[i];
      real prob_cat1;
      // Use subject-specific parameters derived in transformed parameters block
      // Note: Need to access subj_w, subj_c, subj_bias which are available here
      vector[N_features] current_w = subj_w[s];
      real current_c = subj_c[s]; // Use derived original c
      real current_bias = subj_bias[s]; // Use derived bias
      int current_n_mem = n_memory[s];
      int has_cat0 = 0; int has_cat1 = 0;

      // Check memory state
      if (current_n_mem > 0) {
         for (k in 1:current_n_mem) {
           if (memory_cat[s, k] == 0) has_cat0 = 1;
           if (memory_cat[s, k] == 1) has_cat1 = 1;
         }
      }

      // Calculate choice probability (identical logic to model block)
      if (current_n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) {
        prob_cat1 = current_bias;
      } else {
        vector[current_n_mem] current_similarities;
        for (e in 1:current_n_mem) {
          real dist_val = 0;
          for (f in 1:N_features) {
             dist_val += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]);
          }
          // Use derived original current_c
          current_similarities[e] = exp(-current_c * dist_val);
        }
        real sum_sim_cat1 = 0; real sum_sim_cat0 = 0;
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) sum_sim_cat1 += current_similarities[e];
          else sum_sim_cat0 += current_similarities[e];
        }
        real numerator = current_bias * sum_sim_cat1;
        real denominator = numerator + (1 - current_bias) * sum_sim_cat0;
        if (denominator > 1e-9) prob_cat1 = numerator / denominator;
        else prob_cat1 = current_bias;
        prob_cat1 = fmax(1e-9, fmin(1.0 - 1e-9, prob_cat1));
      }

      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1); // Log likelihood

      // Update memory state (identical logic to model block)
      int current_trial_idx = n_memory[s] + 1;
      if (current_trial_idx <= N_trials) {
          for(f in 1:N_features) memory_obs[s, current_trial_idx, f] = obs[i, f];
          memory_cat[s, current_trial_idx] = cat_feedback[i];
          n_memory[s] = current_trial_idx;
      }
    }
  }
}

