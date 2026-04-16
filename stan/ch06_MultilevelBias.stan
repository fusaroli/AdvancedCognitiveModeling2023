
/* Multilevel Bernoulli Model
 * This model infers agent-specific choice biases from sequences of binary choices (0/1)
 * The model assumes each agent has their own bias (theta) drawn from a population distribution
 */
functions {
  // Generate random numbers from truncated normal distribution
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}
data {
  int<lower=1> trials;                // Number of trials per agent
  int<lower=1> agents;                // Number of agents
  array[trials, agents] int<lower=0, upper=1> h;  // Choice data: 0 or 1 for each trial/agent
}
parameters {
  real thetaM;                      // Population-level mean bias (log-odds scale)
  real<lower=0> thetaSD;            // Population-level SD of bias
  array[agents] real theta;         // Agent-specific biases (log-odds scale)
}
model {
  // Population-level priors
  target += normal_lpdf(thetaM | 0, 1);        // Prior for population mean 
  target += normal_lpdf(thetaSD | 0, 0.3)      // Half-normal prior for population SD
    - normal_lccdf(0 | 0, 0.3);               // Adjustment for truncation at 0
  // Agent-level model
  target += normal_lpdf(theta | thetaM, thetaSD);    // Agent biases drawn from population
  // Likelihood for observed choices
  for (i in 1:agents) {
    target += bernoulli_logit_lpmf(h[,i] | theta[i]);  // Choice likelihood
  }
}
generated quantities {
  // Prior predictive samples
  real thetaM_prior = normal_rng(0, 1);
  real<lower=0> thetaSD_prior = normal_lb_rng(0, 0.3, 0);
  real<lower=0, upper=1> theta_prior = inv_logit(normal_rng(thetaM_prior, thetaSD_prior));
  // Posterior predictive samples
  real<lower=0, upper=1> theta_posterior = inv_logit(normal_rng(thetaM, thetaSD));
  // Predictive simulations
  int<lower=0, upper=trials> prior_preds = binomial_rng(trials, inv_logit(thetaM_prior));
  int<lower=0, upper=trials> posterior_preds = binomial_rng(trials, inv_logit(thetaM));
  // Convert parameters to probability scale for easier interpretation
  real<lower=0, upper=1> thetaM_prob = inv_logit(thetaM);
}

