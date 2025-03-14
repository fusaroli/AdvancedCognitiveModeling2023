
// Multilevel Simple Beta-Binomial Model
// This model assumes equal weights for evidence sources but allows 
// individual variation in baseline tendencies

data {
  int<lower=1> N;                        // Total number of observations
  int<lower=1> J;                        // Number of subjects
  array[N] int<lower=1, upper=J> agent_id;  // Agent ID for each observation
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence
}

parameters {
  // Population-level (fixed) effects
  real<lower=0, upper=1> population_theta; // Population average probability
  
  // Individual-level (random) effects
  real<lower=0> tau;                      // Between-subject variability
  vector[J] z;                            // Standardized individual deviations
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0, upper=1>[J] theta;      // Individual probabilities
  
  // Non-centered parameterization
  for (j in 1:J) {
    // Transform z to probability space via logit
    real logit_population = logit(population_theta);
    real logit_individual = logit_population + z[j] * tau;
    theta[j] = inv_logit(logit_individual);
  }
}

model {
  // Priors for population parameters
  population_theta ~ beta(1, 1);        // Uniform prior
  tau ~ exponential(1);                 // Prior for between-subject variability
  z ~ std_normal();                     // Prior for standardized random effects
  
  // Likelihood
  for (i in 1:N) {
    // Calculate combined theta for this specific decision
    real p_blue1 = (blue1[i] + 1.0) / (total1[i] + 2.0); // Beta-binomial posterior
    real p_blue2 = (blue2[i] + 1.0) / (total2[i] + 2.0); // Beta-binomial posterior
    
    // Simple average of the two probabilities
    real combined_theta = (p_blue1 + p_blue2) / 2.0;
    
    // Individual adjustment to the combined theta
    real adjusted_theta = theta[agent_id[i]] * combined_theta;
    
    // Bound probability between 0 and 1
    adjusted_theta = fmin(fmax(adjusted_theta, 0.001), 0.999);
    
    // Model choice
    choice[i] ~ bernoulli(adjusted_theta);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Population and individual predictions
  array[N] int pred_choice;
  
  for (i in 1:N) {
    // Calculate the same combined theta as in the model block
    real p_blue1 = (blue1[i] + 1.0) / (total1[i] + 2.0);
    real p_blue2 = (blue2[i] + 1.0) / (total2[i] + 2.0);
    real combined_theta = (p_blue1 + p_blue2) / 2.0;
    real adjusted_theta = theta[agent_id[i]] * combined_theta;
    adjusted_theta = fmin(fmax(adjusted_theta, 0.001), 0.999);
    
    // Generate predictions
    pred_choice[i] = bernoulli_rng(adjusted_theta);
    
    // Calculate log likelihood
    log_lik[i] = bernoulli_lpmf(choice[i] | adjusted_theta);
  }
}

