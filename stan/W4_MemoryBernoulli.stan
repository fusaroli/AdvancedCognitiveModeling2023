
// The input (data) for the model. n of trials and h for (right and left) hand
data {
 int<lower=1> n;
 array[n] int h;
 vector<lower=0, upper=1>[n] memory; // here we add the new parameter
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
}

// The model to be estimated. 
model {
  // The prior for theta is a uniform distribution between 0 and 1
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(beta | 0, .5);
  
  // The model consists of a binomial distributions with a rate theta
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
  
  bias_prior = normal_rng(0, 0.1);
  beta_prior = normal_rng(0, 0.5);
  prior_preds5 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.5));
  prior_preds7 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.7));
  prior_preds9 = binomial_rng(n, inv_logit(bias_prior + beta_prior * 0.9));
  post_preds5 = binomial_rng(n, inv_logit(bias + beta * 0.5));
  post_preds7 = binomial_rng(n, inv_logit(bias + beta * 0.7));
  post_preds9 = binomial_rng(n, inv_logit(bias + beta * 0.9));

}


