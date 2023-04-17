
// The input data 
data {
 int<lower=1> n;
 array[n] int c;
 vector[n] outcome_hand; 
 real prior_mean_bias;
 real<lower=0> prior_sd_bias;
 real prior_mean_follow_bias;
 real<lower=0> prior_sd_follow_bias;
 
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real follow_bias;
}

// The model
model {
  // priors
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias);
  target += normal_lpdf(follow_bias | prior_mean_follow_bias, prior_sd_follow_bias);
  
  // model
  target += bernoulli_logit_lpmf(c | bias + follow_bias * outcome_hand);
}

// Saving Priors & posteriors
generated quantities{
  real <lower=0, upper=1> bias_prior;
  real <lower=0, upper=1> bias_posterior;
   
  real <lower=0, upper=1> follow_bias_prior;
  real <lower=0, upper=1> follow_bias_posterior;
  
  int <lower=0, upper=n> prior_preds;
  int <lower=0, upper=n> posterior_preds;
  
  array[n] int <lower=0, upper=n> prior_choice;
  array[n] int <lower=0, upper=n> posterior_choice;
  
  bias_prior = inv_logit(normal_rng(0,1));
  bias_posterior = inv_logit(bias);
  
  follow_bias_prior = inv_logit(normal_rng(0,1));
  follow_bias_posterior = inv_logit(follow_bias);
  
  for (i in 1:n){
    prior_choice[i] = binomial_rng(1, inv_logit(bias_prior + follow_bias_prior * outcome_hand[i]));
    posterior_choice[i] = binomial_rng(1, inv_logit(bias_posterior + follow_bias_posterior * outcome_hand[i]));
  }
  
  prior_preds = sum(prior_choice);
  posterior_preds = sum(posterior_choice);
  
}

