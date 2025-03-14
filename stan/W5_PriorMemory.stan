
  // The input (data) for the model
data {
  int<lower=1> n;
  array[n] int h;
  array[n] int other;
  real prior_mean_bias;
  real<lower=0> prior_sd_bias;
  real prior_mean_beta;
  real<lower=0> prior_sd_beta;
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
      memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / trial);
      if (memory[trial + 1] == 0){memory[trial + 1] = 0.01;}
      if (memory[trial + 1] == 1){memory[trial + 1] = 0.99;}
    }
  }
}

// The model to be estimated. 
model {
  // The priors 
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias);
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta);
  
  // The model
  target += bernoulli_logit_lpmf(h | bias + beta * logit(memory));
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
  
  bias_prior = normal_rng(prior_mean_bias, prior_sd_bias);
  beta_prior = normal_rng(prior_mean_beta, prior_sd_beta);
  prior_preds5 = binomial_rng(n, inv_logit(bias_prior + beta_prior * logit(0.5)));
  prior_preds7 = binomial_rng(n, inv_logit(bias_prior + beta_prior * logit(0.7)));
  prior_preds9 = binomial_rng(n, inv_logit(bias_prior + beta_prior * logit(0.9)));
  post_preds5 = binomial_rng(n, inv_logit(bias + beta * logit(0.5)));
  post_preds7 = binomial_rng(n, inv_logit(bias + beta * logit(0.7)));
  post_preds9 = binomial_rng(n, inv_logit(bias + beta * logit(0.9)));
  
}

