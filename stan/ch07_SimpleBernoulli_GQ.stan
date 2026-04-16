
data {
  int<lower=1> n;                      // Number of trials (e.g., 120)
  array[n] int<lower=0, upper=1> h;    // Observed choices (vector of 0s and 1s)
  real prior_theta_m;
  real<lower = 0> prior_theta_sd;
}
parameters {
  real theta_logit; // Represents the log-odds of choosing option 1 (right)
}
model {
  target += normal_lpdf(theta_logit | prior_theta_m, prior_theta_sd); // prior
  target += bernoulli_logit_lpmf(h | theta_logit); // likelihood
}

generated quantities {
  // Create the prior for theta
  real theta_logit_prior = normal_rng(prior_theta_m, prior_theta_sd); 
  // --- Parameter Transformation ---
  real theta_prior = inv_logit(theta_logit_prior); 
  real<lower=0, upper=1> theta = inv_logit(theta_logit);

  // --- Prior Predictive Check Simulation ---
  // Simulate data based *only* on the prior distribution.
  array[n] int h_prior_rep = bernoulli_logit_rng(rep_vector(theta_logit_prior, n));
  // Calculate a summary statistic for this PRIOR replicated dataset.
   int<lower=0, upper=n> prior_rep_sum = sum(h_prior_rep);

  // --- Posterior Predictive Check Simulation ---
  // Simulate data based on the *posterior* distribution of the parameter.
  array[n] int h_post_rep = bernoulli_logit_rng(rep_vector(theta_logit, n));
  // Calculate a summary statistic for this posterior dataset.
  int<lower=0, upper=n> post_rep_sum = sum(h_post_rep); 
  
  // --- Log-likelihood ---
  vector[n] log_lik;
  real lprior;
  // log likelihood
  for (i in 1:n) log_lik[i] =  bernoulli_logit_lpmf(h[i] | theta_logit);
  // joint log prior
  lprior = normal_lpdf(theta_logit | prior_theta_m, prior_theta_sd);
}

