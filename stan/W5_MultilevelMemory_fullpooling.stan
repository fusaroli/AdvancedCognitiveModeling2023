

// The input (data) for the model. n of trials and h of heads
data {
 int<lower = 1> trials;
 int<lower = 1> agents;
 array[trials, agents] int h;
 array[trials, agents] int other;
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real beta;
}


transformed parameters {
  array[trials, agents] real memory;
  
  for (agent in 1:agents){
    for (trial in 1:trials){
      if (trial == 1) {
        memory[trial, agent] = 0.5;
      } 
      if (trial < trials){
        memory[trial + 1, agent] = memory[trial, agent] + ((other[trial, agent] - memory[trial, agent]) / trial);
        if (memory[trial + 1, agent] == 0){memory[trial + 1, agent] = 0.01;}
        if (memory[trial + 1, agent] == 1){memory[trial + 1, agent] = 0.99;}
      }
    }
  }
}

// The model to be estimated. 
model {
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(beta | 0, 1);

  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial, agent] | bias +  memory[trial, agent] * beta);
    }
  }
    
}


generated quantities{
   real bias_prior;
   real beta_prior;
   
   int<lower=0, upper = trials> prior_preds0;
   int<lower=0, upper = trials> prior_preds1;
   int<lower=0, upper = trials> prior_preds2;
   int<lower=0, upper = trials> posterior_preds0;
   int<lower=0, upper = trials> posterior_preds1;
   int<lower=0, upper = trials> posterior_preds2;
   
   bias_prior = normal_rng(0,1);
   beta_prior = normal_rng(0,1);
   
   prior_preds0 = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
   prior_preds1 = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
   prior_preds2 = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
   
   posterior_preds0 = binomial_rng(trials, inv_logit(bias +  0 * (beta)));
   posterior_preds1 = binomial_rng(trials, inv_logit(bias +  1 * (beta)));
   posterior_preds2 = binomial_rng(trials, inv_logit(bias +  2 * (beta)));
  
}

