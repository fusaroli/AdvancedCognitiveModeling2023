
// Generalized Context Model (GCM) - Single Subject (Revised for constrained c)

data {
  int<lower=1> ntrials;       // Number of trials
  int<lower=1> nfeatures;     // Number of stimulus features
  
  // Data for each trial
  array[ntrials] int<lower=0, upper=1> y;       // Observed choices (0 or 1)
  array[ntrials, nfeatures] real obs; // Stimulus features for current trial
  array[ntrials] int<lower=0, upper=1> cat_feedback; // True category feedback
  
  // Priors (passed as data)
  vector[nfeatures] w_prior_alpha; // Dirichlet concentration parameters for weights
  // Prior for c_logit_scaled: normal(mean, sd)
  array[2] real c_logit_scaled_prior_params; 
  array[2] real bias_prior_params; // Parameters for bias prior (e.g., beta(alpha, beta))
  
  // Define upper bound for c
  real<lower=0> C_UPPER_BOUND; // e.g., pass 10.0 from R
}

transformed data {
  // Pre-compute indices for categories to speed up calculations
  array[ntrials] int<lower=0, upper=1> cat_feedback_cat0; // Indicator for category 0 feedback
  array[ntrials] int<lower=0, upper=1> cat_feedback_cat1; // Indicator for category 1 feedback
  for (i in 1:ntrials) {
    cat_feedback_cat0[i] = 1 - cat_feedback[i];
    cat_feedback_cat1[i] = cat_feedback[i];
  }
}

parameters {
  simplex[nfeatures] w;       // Attention weights (sum to 1)
  // real<lower=0> c;         // *** REMOVED ***
  real c_logit_scaled;      // Scaled logit transformed sensitivity
  real<lower=0, upper=1> bias; // Response bias (0 to 1)
}

transformed parameters {
  // Calculate c from the transformed parameter
  // c = L + (U - L) * inv_logit(c_logit_scaled)
  // Since L=0, c = U * inv_logit(c_logit_scaled)
  real<lower=0, upper=C_UPPER_BOUND> c = C_UPPER_BOUND * inv_logit(c_logit_scaled); 
}

model {
  // Priors
  target += dirichlet_lpdf(w | w_prior_alpha);          // Prior on attention weights
  // Prior on the TRANSFORMED sensitivity parameter
  target += normal_lpdf(c_logit_scaled | c_logit_scaled_prior_params[1], c_logit_scaled_prior_params[2]); 
  target += beta_lpdf(bias | bias_prior_params[1], bias_prior_params[2]); // Prior on bias

  // Likelihood calculation trial-by-trial
  { // Local block for memory variables
    array[ntrials, nfeatures] real memory_obs; // Store observed stimuli features
    array[ntrials] int memory_cat;        // Store observed category feedback
    int n_memory = 0;                     // Counter for items in memory

    for (i in 1:ntrials) {
      real prob_cat1; // Probability of choosing category 1

      // Check if memory contains exemplars from both categories
      int has_cat0 = 0;
      int has_cat1 = 0;
      if (n_memory > 0) {
        for (k in 1:n_memory) {
          if (memory_cat[k] == 0) has_cat0 = 1;
          if (memory_cat[k] == 1) has_cat1 = 1;
        }
      }

      if (n_memory == 0 || has_cat0 == 0 || has_cat1 == 0) {
        // Cold start or missing category: use bias
        prob_cat1 = bias;
      } else {
        // Calculate similarity to all stored exemplars
        vector[n_memory] similarities;
        for (e in 1:n_memory) {
          real dist_val = 0; // Changed from dist_sq_sum for clarity
          // City-block distance (r=1) calculation
          for (f in 1:nfeatures) {
             dist_val += w[f] * abs(obs[i, f] - memory_obs[e, f]);
          }
          // Similarity calculation using the transformed c
          similarities[e] = exp(-c * dist_val); 
        }

        // Calculate summed similarity to each category
        real sum_sim_cat1 = 0;
        real sum_sim_cat0 = 0;
        for (e in 1:n_memory) {
          if (memory_cat[e] == 1) {
            sum_sim_cat1 += similarities[e];
          } else {
            sum_sim_cat0 += similarities[e];
          }
        }
        
        // Calculate probability of choosing category 1
        real numerator = bias * sum_sim_cat1;
        real denominator = numerator + (1 - bias) * sum_sim_cat0;

        if (denominator > 1e-9) { // Avoid division by zero
          prob_cat1 = numerator / denominator;
        } else {
          prob_cat1 = bias; // Default to bias if no similarity
        }
        
        // Clamp probability
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1));
      }
      
      // Add likelihood contribution for the current trial
      target += bernoulli_lpmf(y[i] | prob_cat1);

      // Update memory (after calculating likelihood for trial i)
      n_memory += 1;
      memory_obs[n_memory] = obs[i];
      memory_cat[n_memory] = cat_feedback[i];
    } // End trial loop
  } // End local block
}

generated quantities {
  // Log likelihood for model comparison (optional but recommended)
  vector[ntrials] log_lik;
  
  { // Local block for memory variables (must match model block structure)
    array[ntrials, nfeatures] real memory_obs; 
    array[ntrials] int memory_cat;        
    int n_memory = 0;                     

    for (i in 1:ntrials) {
      real prob_cat1; 
      int has_cat0 = 0;
      int has_cat1 = 0;
      if (n_memory > 0) {
        for (k in 1:n_memory) {
          if (memory_cat[k] == 0) has_cat0 = 1;
          if (memory_cat[k] == 1) has_cat1 = 1;
        }
      }

      if (n_memory == 0 || has_cat0 == 0 || has_cat1 == 0) {
        prob_cat1 = bias;
      } else {
        vector[n_memory] similarities;
        for (e in 1:n_memory) {
          real dist_val = 0; 
          for (f in 1:nfeatures) {
             dist_val += w[f] * abs(obs[i, f] - memory_obs[e, f]);
          }
          // Use the transformed c here as well
          similarities[e] = exp(-c * dist_val); 
        }
        real sum_sim_cat1 = 0;
        real sum_sim_cat0 = 0;
        for (e in 1:n_memory) {
          if (memory_cat[e] == 1) {
            sum_sim_cat1 += similarities[e];
          } else {
            sum_sim_cat0 += similarities[e];
          }
        }
        real numerator = bias * sum_sim_cat1;
        real denominator = numerator + (1 - bias) * sum_sim_cat0;
        if (denominator > 1e-9) { 
          prob_cat1 = numerator / denominator;
        } else {
          prob_cat1 = bias; 
        }
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1));
      }
      
      // Calculate log likelihood for trial i
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1);

      // Update memory
      n_memory += 1;
      memory_obs[n_memory] = obs[i];
      memory_cat[n_memory] = cat_feedback[i];
    } 
  } 
}

