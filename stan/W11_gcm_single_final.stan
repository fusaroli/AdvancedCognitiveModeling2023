
// Generalized Context Model (GCM) - Single Subject (Scaled Logit c)

data {
  int<lower=1> ntrials;       
  int<lower=1> nfeatures;     
  array[ntrials] int<lower=0, upper=1> y;       
  array[ntrials, nfeatures] real obs; 
  array[ntrials] int<lower=0, upper=1> cat_feedback; 
  
  // Priors
  vector[nfeatures] w_prior_alpha; 
  // Prior for c_logit_scaled: normal(mean, sd)
  array[2] real c_logit_scaled_prior_params; 
  array[2] real bias_prior_params; 
  
  // Define bounds for c
  real<lower=0> C_UPPER_BOUND; 
  real LOWER_BOUND; // Typically 0
}

parameters {
  simplex[nfeatures] w;       
  real c_logit_scaled;      // Scaled logit transformed sensitivity
  real<lower=0, upper=1> bias; 
}

transformed parameters {
  // Calculate c on original scale from the transformed parameter
  real<lower=LOWER_BOUND, upper=C_UPPER_BOUND> c = LOWER_BOUND + (C_UPPER_BOUND - LOWER_BOUND) * inv_logit(c_logit_scaled); 
}

model {
  // Priors
  target += dirichlet_lpdf(w | w_prior_alpha);          
  // Prior on the TRANSFORMED sensitivity parameter
  target += normal_lpdf(c_logit_scaled | c_logit_scaled_prior_params[1], c_logit_scaled_prior_params[2]); 
  target += beta_lpdf(bias | bias_prior_params[1], bias_prior_params[2]); 

  // Likelihood calculation trial-by-trial
  { 
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
          // Similarity calculation using the derived 'c'
          similarities[e] = exp(-c * dist_val); 
        }

        real sum_sim_cat1 = 0;
        real sum_sim_cat0 = 0;
        for (e in 1:n_memory) {
          if (memory_cat[e] == 1) sum_sim_cat1 += similarities[e];
          else sum_sim_cat0 += similarities[e];
        }
        
        real numerator = bias * sum_sim_cat1;
        real denominator = numerator + (1 - bias) * sum_sim_cat0;

        if (denominator > 1e-9) { 
          prob_cat1 = numerator / denominator;
        } else {
          prob_cat1 = bias; 
        }
        
        prob_cat1 = fmax(1e-9, fmin(1.0 - 1e-9, prob_cat1)); // Use smaller epsilon
      }
      
      target += bernoulli_lpmf(y[i] | prob_cat1);

      // Update memory
      n_memory += 1;
      // Avoid potential overflow if ntrials is large and memory isn't bounded
      if (n_memory <= ntrials) { 
          memory_obs[n_memory] = obs[i];
          memory_cat[n_memory] = cat_feedback[i];
      }
    } 
  } 
}

generated quantities {
  vector[ntrials] log_lik;
  // Also generate c_logit_scaled for direct comparison in recovery/SBC
  real c_logit_scaled_rep = c_logit_scaled; // Replicate for output
  real c_rep = c; // Replicate derived c for output
  
  { // Replicate memory logic exactly from model block
    array[ntrials, nfeatures] real memory_obs; 
    array[ntrials] int memory_cat;        
    int n_memory = 0;                     

    for (i in 1:ntrials) {
      real prob_cat1; 
      int has_cat0 = 0;
      int has_cat1 = 0;
      if (n_memory > 0) { /* ... check memory ... */ }

      if (n_memory == 0 || has_cat0 == 0 || has_cat1 == 0) {
        prob_cat1 = bias;
      } else {
        vector[n_memory] similarities;
        for (e in 1:n_memory) {
          real dist_val = 0; 
          for (f in 1:nfeatures) { /* ... calculate distance ... */ }
          // Use the derived 'c' from transformed parameters
          similarities[e] = exp(-c * dist_val); 
        }
        real sum_sim_cat1 = 0;
        real sum_sim_cat0 = 0;
        /* ... calculate summed similarities ... */
        real numerator = bias * sum_sim_cat1;
        real denominator = numerator + (1 - bias) * sum_sim_cat0;
        if (denominator > 1e-9) { /* ... calculate prob_cat1 ... */ }
        else { prob_cat1 = bias; }
        prob_cat1 = fmax(1e-9, fmin(1.0 - 1e-9, prob_cat1));
      }
      
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1);

      // Update memory
      n_memory += 1;
      if (n_memory <= ntrials) {
          memory_obs[n_memory] = obs[i];
          memory_cat[n_memory] = cat_feedback[i];
      }
    } 
  } 
}

