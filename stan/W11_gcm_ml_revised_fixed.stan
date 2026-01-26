
// Generalized Context Model (GCM) - Multilevel Version
data {
  int<lower=1> N_total;      // Total number of observations across all subjects
  int<lower=1> N_subjects;    // Number of subjects
  int<lower=1> N_trials;      // Max number of trials per subject (used for memory array sizing)
  int<lower=1> N_features;    // Number of stimulus features
  // Data for each observation
  array[N_total] int<lower=1, upper=N_subjects> subj_id; // Subject ID for each obs
  array[N_total] int<lower=0, upper=1> y;               // Observed choices (0 or 1)
  array[N_total, N_features] real obs;                   // Stimulus features for current trial
  array[N_total] int<lower=0, upper=1> cat_feedback;    // True category feedback
  // Priors (passed as data) - Population level
  vector[N_features] pop_w_prior_alpha; // Dirichlet concentration for population weights
  array[2] real pop_c_prior_params;     // Prior for population log(c) mean and sd (e.g., [mean_mean, mean_sd])
  array[2] real pop_c_sd_prior_params; // Prior for population log(c) sd (e.g., [sd_mean, sd_sd] for half-normal or exponential)
  // Priors for logit-bias
  array[2] real pop_logit_bias_prior_params; // Prior for population logit_bias mean and sd (e.g., [mean_mean, mean_sd])
  array[2] real pop_logit_bias_sd_prior_params; // Prior for population logit_bias sd (e.g., [sd_mean, sd_sd] for half-normal or exponential)
  real kappa_prior_rate;                 // Rate for exponential prior on kappa
}
parameters {
  // Population-level parameters
  simplex[N_features] pop_w;          // Population mean attention weights
  real<lower=0> kappa;                // Concentration parameter for weights (>0)
  real pop_log_c_mean;                // Population mean of log(sensitivity)
  real<lower=0> pop_log_c_sd;         // Population sd of log(sensitivity) (>0)
  // Logit-scale bias parameters
  real pop_logit_bias_mean;           // Population mean of logit(bias)
  real<lower=0> pop_logit_bias_sd;    // Population sd of logit(bias) (>0)
  // Individual-level parameters (non-centered parameterization)
  vector[N_subjects] z_log_c;         // Standardized individual deviations for log(c)
  vector[N_subjects] z_logit_bias;    // Standardized individual deviations for logit_bias
  // Individual weights (Dirichlet parameterization)
  array[N_subjects] simplex[N_features] subj_w; // Individual weights
}
transformed parameters {
  // Transform individual parameters back to natural scale
  vector<lower=0>[N_subjects] subj_c;      // Individual sensitivity
  vector<lower=0, upper=1>[N_subjects] subj_bias; // Individual bias
  for (j in 1:N_subjects) {
    subj_c[j] = exp(pop_log_c_mean + z_log_c[j] * pop_log_c_sd);
    // Use inverse logit for bias based on logit-normal hierarchy
    subj_bias[j] = inv_logit(pop_logit_bias_mean + z_logit_bias[j] * pop_logit_bias_sd);
  }
}
model {
  // Priors for population-level parameters
  target += dirichlet_lpdf(pop_w | pop_w_prior_alpha);
  target += exponential_lpdf(kappa | kappa_prior_rate); // Or other appropriate prior like gamma
  target += normal_lpdf(pop_log_c_mean | pop_c_prior_params[1], pop_c_prior_params[2]);
  target += normal_lpdf(pop_log_c_sd | pop_c_sd_prior_params[1], pop_c_sd_prior_params[2]); // Example: half-normal if [1]=0
  // Priors for logit-bias parameters
  target += normal_lpdf(pop_logit_bias_mean | pop_logit_bias_prior_params[1], pop_logit_bias_prior_params[2]);
  target += normal_lpdf(pop_logit_bias_sd | pop_logit_bias_sd_prior_params[1], pop_logit_bias_sd_prior_params[2]); // Example: half-normal if [1]=0
  // Priors for individual deviations (non-centered)
  target += std_normal_lpdf(z_log_c);
  target += std_normal_lpdf(z_logit_bias); // Prior for standardized logit_bias deviations
  // Hierarchical prior for individual weights
  // This links individual weights to the population parameters
  for (j in 1:N_subjects) {
    target += dirichlet_lpdf(subj_w[j] | kappa * pop_w);
  }
  // Likelihood calculation (iterating through all observations)
  { // Local block for memory management (memory arrays are local to this block)
    // These arrays store the memory for each subject dynamically as we loop through trials
    // Assumes N_trials is large enough to hold max trials for any subject
    array[N_subjects, N_trials, N_features] real memory_obs;
    array[N_subjects, N_trials] int memory_cat;
    array[N_subjects] int n_memory; // Counter for items in memory per subject
    // Initialize memory counters
    for (j in 1:N_subjects) {
      n_memory[j] = 0;
    }
    // Main loop through observations
    // IMPORTANT: Assumes data is ordered by subject and then by trial for memory updating to work correctly.
    for (i in 1:N_total) {
      // ------------ BEGIN DECLARATIONS (Must be FIRST in loop block) ------------
      // Declare ONLY variables needed in the immediate scope or unconditionally used
      int s;                  // Current subject ID
      real prob_cat1;         // Probability of choosing category 1
      simplex[N_features] current_w; // Subject's weights
      real current_c;         // Subject's sensitivity
      real current_bias;      // Subject's bias
      int current_n_mem;      // Number of items currently in this subject's memory
      int has_cat0;           // Flag: does memory contain category 0 exemplars?
      int has_cat1;           // Flag: does memory contain category 1 exemplars?
      int current_trial_idx;  // Index for updating memory
      // --- Intermediate calculation variables are now declared INSIDE the 'else' block ---
      // ------------ END DECLARATIONS ------------
      // ------------ BEGIN EXECUTABLE STATEMENTS (Assignments, Logic, etc.) ------------
      s = subj_id[i]; // Get current subject ID (ASSIGNMENT)
      current_w = subj_w[s]; // ASSIGNMENT
      current_c = subj_c[s]; // ASSIGNMENT
      current_bias = subj_bias[s]; // ASSIGNMENT
      current_n_mem = n_memory[s]; // ASSIGNMENT
      has_cat0 = 0; // ASSIGNMENT
      has_cat1 = 0; // ASSIGNMENT
      // Check if memory contains exemplars from both categories for this subject
      if (current_n_mem > 0) { // CONDITIONAL statement
        // Check memory contents (Loop and Conditional)
        for (k in 1:current_n_mem) {
          if (memory_cat[s, k] == 0) has_cat0 = 1;
          if (memory_cat[s, k] == 1) has_cat1 = 1;
          if (has_cat0 == 1 && has_cat1 == 1) break;
        }
      }
      // Calculate probability based on GCM
      if (current_n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) { // CONDITIONAL statement
        prob_cat1 = current_bias; // ASSIGNMENT
      } else {
        // --- Declare similarity vector and intermediate variables INSIDE the 'else' block ---
        vector[current_n_mem] similarities; // DECLARATION **inside** inner block
        real sum_sim_cat1; // DECLARATION **inside** inner block
        real sum_sim_cat0; // DECLARATION **inside** inner block
        real numerator;    // DECLARATION **inside** inner block
        real denominator;  // DECLARATION **inside** inner block
        // Calculate similarity to all items in memory (Loop)
        for (e in 1:current_n_mem) {
          real dist_w_sum = 0; // DECLARATION inside loop - Allowed
          for (f in 1:N_features) {
             dist_w_sum += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]); // Calculation
          }
          similarities[e] = exp(-current_c * fmax(0.0, dist_w_sum)); // ASSIGNMENT
        }
        // Sum similarities by category
        sum_sim_cat1 = 0; // ASSIGNMENT
        sum_sim_cat0 = 0; // ASSIGNMENT
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) {
            sum_sim_cat1 += similarities[e]; // Calculation/Assignment
          } else {
            sum_sim_cat0 += similarities[e]; // Calculation/Assignment
          }
        }
        // Calculate probability P(Cat1 | obs)
        numerator = current_bias * sum_sim_cat1; // ASSIGNMENT
        denominator = numerator + (1 - current_bias) * sum_sim_cat0; // ASSIGNMENT
        if (denominator > 1e-9) { // CONDITIONAL statement
          prob_cat1 = numerator / denominator; // ASSIGNMENT
        } else {
          prob_cat1 = current_bias; // ASSIGNMENT
        }
        // Clamp probability
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1)); // ASSIGNMENT/Function Call
      }
      // Add likelihood contribution for the current observation
      target += bernoulli_lpmf(y[i] | prob_cat1); // Stan Function Call
      // Update memory for this subject
      current_trial_idx = n_memory[s] + 1; // ASSIGNMENT
      if (current_trial_idx <= N_trials) { // CONDITIONAL statement
          memory_obs[s, current_trial_idx] = obs[i]; // ASSIGNMENT
          memory_cat[s, current_trial_idx] = cat_feedback[i]; // ASSIGNMENT
          n_memory[s] = current_trial_idx; // ASSIGNMENT
      } else {
          // Optional: print("Warning: Memory overflow for subject ", s);
      }
      // ------------ END EXECUTABLE STATEMENTS ------------
    } // End observation loop (i)
  } // End local block for memory arrays
}
generated quantities {
  vector[N_total] log_lik; // Log likelihood for each observation
  { // Replicate memory logic from model block
    // These need to be re-declared and re-populated here
    array[N_subjects, N_trials, N_features] real memory_obs;
    array[N_subjects, N_trials] int memory_cat;
    array[N_subjects] int n_memory;
    for (j in 1:N_subjects) { n_memory[j] = 0; }
    for (i in 1:N_total) {
      // *** Declare all local variables for the loop iteration first ***
      int s;
      real prob_cat1;
      simplex[N_features] current_w;
      real current_c;
      real current_bias;
      int current_n_mem;
      int has_cat0;
      int has_cat1;
      int current_trial_idx;
      // *** Now assign values and perform logic ***
      s = subj_id[i];
      current_w = subj_w[s]; // Uses the sampled parameter values
      current_c = subj_c[s];
      current_bias = subj_bias[s];
      current_n_mem = n_memory[s];
      has_cat0 = 0;
      has_cat1 = 0;
      // --- Replicate the probability calculation logic exactly from the model block ---
      if (current_n_mem > 0) {
        for (k in 1:current_n_mem) {
          if (memory_cat[s, k] == 0) has_cat0 = 1;
          if (memory_cat[s, k] == 1) has_cat1 = 1;
            if (has_cat0 == 1 && has_cat1 == 1) break;
        }
      }
      if (current_n_mem == 0 || has_cat0 == 0 || has_cat1 == 0) {
        prob_cat1 = current_bias;
      } else {
        vector[current_n_mem] similarities;
        real sum_sim_cat1; // Declared inside 'else'
        real sum_sim_cat0; // Declared inside 'else'
        real numerator;    // Declared inside 'else'
        real denominator;  // Declared inside 'else'
        for (e in 1:current_n_mem) {
          real dist_w_sum = 0;
          for (f in 1:N_features) {
              dist_w_sum += current_w[f] * abs(obs[i, f] - memory_obs[s, e, f]);
          }
          similarities[e] = exp(-current_c * fmax(0.0, dist_w_sum));
        }
        sum_sim_cat1 = 0; sum_sim_cat0 = 0;
        for (e in 1:current_n_mem) {
          if (memory_cat[s, e] == 1) sum_sim_cat1 += similarities[e];
          else sum_sim_cat0 += similarities[e];
        }
        numerator = current_bias * sum_sim_cat1;
        denominator = numerator + (1 - current_bias) * sum_sim_cat0;
        if (denominator > 1e-9) prob_cat1 = numerator / denominator;
        else prob_cat1 = current_bias;
        prob_cat1 = fmax(1e-6, fmin(1.0 - 1e-6, prob_cat1));
      }
      // --- End of replicated probability calculation ---
      // Calculate log likelihood for this observation
      log_lik[i] = bernoulli_lpmf(y[i] | prob_cat1);
      // Update memory (must mirror the model block exactly)
      current_trial_idx = n_memory[s] + 1;
      if (current_trial_idx <= N_trials) {
          memory_obs[s, current_trial_idx] = obs[i];
          memory_cat[s, current_trial_idx] = cat_feedback[i];
          n_memory[s] = current_trial_idx;
      }
    } // End observation loop (i)
  } // End local block
}

