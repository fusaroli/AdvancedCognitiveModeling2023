
  // The input (data) for the model
data {
  int<lower=1> n;
  array[n] int h;
  real prior_mean_bias;
  real<lower=0> prior_sd_bias;
  real prior_mean_beta;
  real<lower=0> prior_sd_beta;
  vector<lower=0, upper=1>[n] memory; // here we add the new parameter
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
}

// The model to be estimated. 
model {
  // The priors 
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias);
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta);
  
  // The model
  target += bernoulli_logit_lpmf(h | bias + beta * memory);
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
  prior_preds5 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.5));
  prior_preds7 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.7));
  prior_preds9 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.9));
  post_preds5 = binomial_rng(n, inv_logit(bias + beta * 0.5));
  post_preds7 = binomial_rng(n, inv_logit(bias + beta * 0.7));
  post_preds9 = binomial_rng(n, inv_logit(bias + beta * 0.9));
  
}

