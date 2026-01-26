
// Generalized Context Model (GCM) - Multilevel Version (Revised for constrained c)

data {
  int<lower=1> N_total;       // Total number of observations across all subjects
  int<lower=1> N_subjects;    // Number of subjects
  int<lower=1> N_trials;      // Max number of trials per subject (used for memory array sizing)
  int<lower=1> N_features;    // Number of stimulus features
  
  // Data for each observation
  array[N_total] int<lower=1, upper=N_subjects> subj_id; // Subject ID for each obs
  array[N_total] int<lower=0, upper=1> y;         // Observed choices (0 or 1)
  array[N_total, N_features] real obs;           // Stimulus features for current trial
  array[N_total] int<lower=0, upper=1> cat_feedback; // True category feedback
  
  // Priors (passed as data) - Population level
  vector[N_features] pop_w_prior_alpha; // Dirichlet concentration for population weights
  // Prior for population c_logit_scaled mean: normal(mean, sd)
  array[2] real pop_c_logit_scaled_mean_prior_params; 
  // Prior for population c_logit_scaled sd: e.g., normal(0, sd) T[0,] or exponential(rate)
  array[2] real pop_c_logit_scaled_sd_prior_params; // e.g., {0, 1} for Normal(0,1) T[0,]; {1} for Exp(1)
  int<lower=0,upper=1> pop_c_logit_scaled_sd_prior_type; // 0=Normal T[0,], 1=Exponential
  
  array[2] real pop_bias_prior_params; // Prior for population bias mean (e.g., [alpha, beta] for beta)
  real kappa_prior_rate;           // Rate for exponential prior on kappa
  real bias_phi_prior_rate;        // Rate for exponential prior on bias concentration phi
  
  // Define upper bound for c
  real<lower=0> C_UPPER_BOUND; // e.g., pass 10.0 from R
}

parameters {
  // Population-level parameters
  simplex[N_features] pop_w;         // Population mean attention weights
  real<lower=0> kappa;               // Concentration parameter for weights (>0)
  
  // real pop_log_c_mean;           // *** REMOVED ***
  // real<lower=0> pop_log_c_sd;    // *** REMOVED ***
  real pop_c_logit_scaled_mean;      // Population mean of scaled logit(c)
  real<lower=0> pop_c_logit_scaled_sd; // Population sd of scaled logit(c)
  
  real<lower=0, upper=1> pop_bias_mean; // Population mean bias (0-1)
  real<lower=0> pop_bias_phi;        // Population concentration for bias (>0)

  // Individual-level parameters (non-centered parameterization)
  // vector[N_subjects] z_log_c;    // *** REMOVED ***
  vector[N_subjects] z_c_logit_scaled; // Standardized individual deviations for scaled logit(c)
  vector[N_subjects] z_bias;         // Standardized individual deviations for bias (on some scale)
  // Individual weights (centered parameterization - simpler to implement here)
  array[N_subjects] simplex[N_features] subj_w; // Individual weights
}

transformed parameters {
  // Transform individual parameters back to natural scale
  // vector<lower=0>[N_subjects] subj_c;    // *** REMOVED ***
  vector<lower=0, upper=C_UPPER_BOUND>[N_subjects] subj_c; // Individual sensitivity (now bounded)
  vector<lower=0, upper=1>[N_subjects] subj_bias; // Individual bias
  
  for (j in 1:N_subjects) {
    // Sensitivity (from scaled logit)
    // subj_c = L + (U - L) * inv_logit(pop_mean + z * pop_sd)
    // Since L=0, subj_c = U * inv_logit(...)
    subj_c[j] = C_UPPER_BOUND * inv_logit(pop_c_logit_scaled_mean + z_c_logit_scaled[j] * pop_c_logit_scaled_sd);
    
    // Bias (using simplified logit reparam example - check this carefully or use a robust method)
    subj_bias[j] = inv_logit(logit(pop_bias_mean) + z_bias[j] * 1); // Placeholder
  }
}

model {
  // Priors for population-level parameters
  target += dirichlet_lpdf(pop_w | pop_w_prior_alpha);
  target += exponential_lpdf(kappa | kappa_prior_rate);
  
  // Priors for TRANSFORMED sensitivity parameters
  target += normal_lpdf(pop_c_logit_scaled_mean | pop_c_logit_scaled_mean_prior_params[1], pop_c_logit_scaled_mean_prior_params[2]);
  if (pop_c_logit_scaled_sd_prior_type == 0) { // Normal T[0,]
      target += normal_lpdf(pop_c_logit_scaled_sd | pop_c_logit_scaled_sd_prior_params[1], pop_c_logit_scaled_sd_prior_params[2]) - normal_lccdf(0 | pop_c_logit_scaled_sd_prior_params[1], pop_c_logit_scaled_sd_prior_params[2]); 
  } else { // Exponential
      target += exponential_lpdf(pop_c_logit_scaled_sd | pop_c_logit_scaled_sd_prior_params[1]); 
  }

  target += beta_lpdf(pop_bias_mean | pop_bias_prior_params[1], pop_bias_prior_params[2]);
  target += exponential_lpdf(pop_bias_phi | bias_phi_prior_rate);
  
  // Priors for individual deviations (non-centered)
  target += std_normal_lpdf(z_c_logit_scaled); // Prior for sensitivity deviations
  target += std_normal_lpdf(z_bias); // Assumes z_bias is on standard normal scale
  
  // Hierarchical prior for individual weights (centered parameterization)
  for (j in 1:N_subjects) {
    target += dirichlet_lpdf(subj_w[j] | kappa * pop_w); 
  }

  // Likelihood calculation (iterating through all observations)
  { // Local block for memory management 
    // Memory storage needs to be per subject
    array[N_subjects, N_trials, N_features] real memory_obs; 
    array[N_subjects, N_trials] int memory_cat;        
    array[N_subjects] int n_memory; // Counter per subject
    
    int s;              // Current subject ID
    real prob_cat1;     // Probability of choosing category 1
    vector[N_features] current_w; // Subject's weights
    real current_c;       // Subject's sensitivity (now uses bounded subj_c)
    real current_bias;    // Subject's bias
    int current_n_mem;    // Number of items in subject's memory
    int has_cat0;         // Flag: memory has category 0 exemplar?
    int has_cat1;         // Flag: memory has category 1 exemplar?
    real dist_val;        // Distance value
    real sum_sim_cat1;    // Summed similarity cat 1
    real sum_sim_cat0;    // Summed similarity cat 0
    real numerator;       // Numerator for probability calc
    real denominator;     // Denominator for probability calc
    int current_trial_idx;// Index for memory update
    
    for (j in 1:N_subjects) { n_memory[j] = 0; } // Initialize memory counters

    for (i in 1:N_total) {
      has_cat0 = 0; has_cat1 = 0; // Reset flags
      s = subj_id[i]; 
      for (f in 1:N_features) { current_w[f] = subj_w[s][f]; }
      // Use the correctly transformed subj_c[s]
      current_c = subj_c[s]; 
      current_bias = subj_bias[s];
      current_n_mem = n_memory[s];

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
          dist_val = 0; 
          for (f in 1:N_features) {
             dist_val += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]);
          }
          // Similarity calculation uses the bounded current_c
          current_similarities[e] = exp(-current_c * dist_val); 
        }
        sum_sim_cat1 = 0; sum_sim_cat0 = 0;
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) sum_sim_cat1 += current_similarities[e];
          else sum_sim_cat0 += current_similarities[e];
        }
        numerator = current_bias * sum_sim_cat1;
        denominator = numerator + (1 - current_bias) * sum_sim_cat0;
        if (denominator > 1e-9) prob_cat1 = numerator / denominator;
        else prob_cat1 = current_bias; 
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1));
      }
      
      target += bernoulli_lpmf(y[i] | prob_cat1); // Likelihood contribution

      // Update memory
      current_trial_idx = n_memory[s] + 1; 
      if (current_trial_idx <= N_trials) { 
          for(f in 1:N_features) memory_obs[s, current_trial_idx, f] = obs[i, f];
          memory_cat[s, current_trial_idx] = cat_feedback[i];
          n_memory[s] = current_trial_idx;
      } // Else: handle memory overflow if needed
    } // End observation loop
  } // End local block
}

generated quantities {
  vector[N_total] log_lik; // For LOO-CV / model comparison
  
  { // Replicate memory logic from model block 
    array[N_subjects, N_trials, N_features] real memory_obs; 
    array[N_subjects, N_trials] int memory_cat;        
    array[N_subjects] int n_memory; 
    
    int s;              
    real prob_cat1;     
    vector[N_features] current_w; 
    real current_c;       
    real current_bias;    
    int current_n_mem;    
    int has_cat0;         
    int has_cat1;         
    real dist_val;        
    real sum_sim_cat1;    
    real sum_sim_cat0;    
    real numerator;       
    real denominator;     
    int current_trial_idx;
    
    for (j in 1:N_subjects) { n_memory[j] = 0; }

    for (i in 1:N_total) {
      has_cat0 = 0; has_cat1 = 0; // Reset flags
      s = subj_id[i]; 
      for (f in 1:N_features) { current_w[f] = subj_w[s][f]; }
      // Use the correctly transformed subj_c[s]
      current_c = subj_c[s]; 
      current_bias = subj_bias[s];
      current_n_mem = n_memory[s];
      
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
          dist_val = 0; 
          for (f in 1:N_features) {
             dist_val += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]);
          }
          // Use bounded current_c
          current_similarities[e] = exp(-current_c * dist_val); 
        }
        sum_sim_cat1 = 0; sum_sim_cat0 = 0;
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) sum_sim_cat1 += current_similarities[e];
          else sum_sim_cat0 += current_similarities[e];
        }
        numerator = current_bias * sum_sim_cat1;
        denominator = numerator + (1 - current_bias) * sum_sim_cat0;
        if (denominator > 1e-9) prob_cat1 = numerator / denominator;
        else prob_cat1 = current_bias; 
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1));
      }
      
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1); // Log likelihood

      // Update memory state (identical logic to model block)
      current_trial_idx = n_memory[s] + 1; 
      if (current_trial_idx <= N_trials) {
          for(f in 1:N_features) memory_obs[s, current_trial_idx, f] = obs[i, f];
          memory_cat[s, current_trial_idx] = cat_feedback[i];
          n_memory[s] = current_trial_idx;
      }
    } // End observation loop
  } // End local block
}

