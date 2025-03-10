
// Multilevel Biased Agent Model
//
// This model assumes each agent has a fixed bias (theta) that determines
// their probability of choosing option 1 ('right')

functions{
  // Helper function for generating truncated normal random numbers
  real normal_lb_rng(real mu, real sigma, real lb) { 
    real p = normal_cdf(lb | mu, sigma);  // CDF for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // Inverse CDF for value
  }
}

// Input data
data {
 int<lower = 1> trials;         // Number of trials per agent
 int<lower = 1> agents;         // Number of agents
 array[trials, agents] int h;   // Choice data (0/1 for each trial and agent)
}

// Parameters to be estimated
parameters {
  real thetaM;                  // Population mean of bias (log-odds scale)
  real<lower = 0> thetaSD;      // Population SD of bias
  array[agents] real theta;     // Individual agent biases (log-odds scale)
}

// Model definition
model {
  // Population-level priors
  target += normal_lpdf(thetaM | 0, 1);
  
  // Prior for SD with lower bound at zero (half-normal)
  target += normal_lpdf(thetaSD | 0, .3) - normal_lccdf(0 | 0, .3);

  // Individual-level parameters drawn from population distribution
  target += normal_lpdf(theta | thetaM, thetaSD); 
 
  // Likelihood: predict each agent's choices
  for (i in 1:agents)
    target += bernoulli_logit_lpmf(h[,i] | theta[i]);
}

// Additional quantities to calculate
generated quantities{
   // Prior predictive samples
   real thetaM_prior;
   real<lower=0> thetaSD_prior;
   real<lower=0, upper=1> theta_prior;
   real<lower=0, upper=1> theta_posterior;
   
   // Posterior predictive samples
   int<lower=0, upper = trials> prior_preds;
   int<lower=0, upper = trials> posterior_preds;
   
   // Log-likelihood for each observation (crucial for model comparison)
   array[trials, agents] real log_lik;
   
   // Generate prior samples
   thetaM_prior = normal_rng(0, 1);
   thetaSD_prior = normal_lb_rng(0, 0.3, 0);
   theta_prior = inv_logit(normal_rng(thetaM_prior, thetaSD_prior));
   theta_posterior = inv_logit(normal_rng(thetaM, thetaSD));
   
   // Generate predictions from prior and posterior
   prior_preds = binomial_rng(trials, inv_logit(thetaM_prior));
   posterior_preds = binomial_rng(trials, inv_logit(thetaM));
   
   // Calculate log-likelihood for each observation
   for (i in 1:agents){
    for (t in 1:trials){
      log_lik[t, i] = bernoulli_logit_lpmf(h[t, i] | theta[i]);
    }
   }
}

