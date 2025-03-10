
// Multilevel Biased Agent Model with CV support
//
// This model has additional structures to handle test data for cross-validation

functions{
  real normal_lb_rng(real mu, real sigma, real lb) { 
    real p = normal_cdf(lb | mu, sigma);  // CDF for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // Inverse CDF for value
  }
}

// Input data - now includes both training and test data
data {
 int<lower = 1> trials;              // Number of trials per agent
 int<lower = 1> agents;              // Number of training agents
 array[trials, agents] int h;        // Training choice data
 
 int<lower = 1> agents_test;         // Number of test agents
 array[trials, agents_test] int h_test;  // Test choice data
}

// Parameters to be estimated
parameters {
  real thetaM;                       // Population mean of bias
  real<lower = 0> thetaSD;           // Population SD of bias
  array[agents] real theta;          // Individual agent biases
}

// Model definition - trained only on training data
model {
  // Population-level priors
  target += normal_lpdf(thetaM | 0, 1);
  target += normal_lpdf(thetaSD | 0, .3) - normal_lccdf(0 | 0, .3);

  // Individual-level parameters
  target += normal_lpdf(theta | thetaM, thetaSD); 
 
  // Likelihood for training data only
  for (i in 1:agents)
    target += bernoulli_logit_lpmf(h[,i] | theta[i]);
}

// Calculate log-likelihood for both training and test data
generated quantities{
   real thetaM_prior;
   real<lower=0> thetaSD_prior;
   real<lower=0, upper=1> theta_prior;
   real<lower=0, upper=1> theta_posterior;
   
   int<lower=0, upper = trials> prior_preds;
   int<lower=0, upper = trials> posterior_preds;
   
   // Log-likelihood for training data
   array[trials, agents] real log_lik;
   
   // Log-likelihood for test data - crucial for cross-validation
   array[trials, agents_test] real log_lik_test;
   
   // Generate prior and posterior samples
   thetaM_prior = normal_rng(0,1);
   thetaSD_prior = normal_lb_rng(0,0.3,0);
   theta_prior = inv_logit(normal_rng(thetaM_prior, thetaSD_prior));
   theta_posterior = inv_logit(normal_rng(thetaM, thetaSD));
   
   prior_preds = binomial_rng(trials, inv_logit(thetaM_prior));
   posterior_preds = binomial_rng(trials, inv_logit(thetaM));
   
   // Calculate log-likelihood for training data
   for (i in 1:agents){
    for (t in 1:trials){
      log_lik[t,i] = bernoulli_logit_lpmf(h[t,i] | theta[i]);
    }
   }
   
   // Calculate log-likelihood for test data
   // Note: We use population-level estimates for prediction
   for (i in 1:agents_test){
    for (t in 1:trials){
      log_lik_test[t,i] = bernoulli_logit_lpmf(h_test[t,i] | thetaM);
    }
  }
}

