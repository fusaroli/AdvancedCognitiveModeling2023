
// The input (data) for the model. n of trials and h for (right and left) hand
data {
 int<lower=1> n;
 array[n] int h;
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
}

transformed parameters{
  vector[n] memory;

  for (trial in 1:n){
  if (trial == 1) {
    memory[trial] = 0.5;
  } 
  if (trial < n){
      memory[trial + 1] = memory[trial] + ((h[trial] - memory[trial]) / trial);
    }
  }
}

// The model to be estimated. 
model {
  // Priors
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  
  // Model, looping to keep track of memory
  for (trial in 1:n) {
    target += bernoulli_logit_lpmf(h[trial] | bias + beta * inv_logit(memory[trial]));
  }
}

generated quantities{
  real bias_prior;
  real beta_prior;
  int<lower=0, upper=n> prior_preds5;
  int<lower=0, upper=n> post_preds5;
  int<lower=0, upper=n> prior_preds7;
  int<lower=0, upper=n> post_preds7;
  int<lower=0, upper=n> prior_preds9;
  int<lower=0, upper=n> post_preds9;
  
  bias_prior = normal_rng(0, 0.1);
  beta_prior = normal_rng(0, 0.5);
  prior_preds5 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.5));
  prior_preds7 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.7));
  prior_preds9 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.9));
  post_preds5 = binomial_rng(n, inv_logit(bias + beta * 0.5));
  post_preds7 = binomial_rng(n, inv_logit(bias + beta * 0.7));
  post_preds9 = binomial_rng(n, inv_logit(bias + beta * 0.9));

}


