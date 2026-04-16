
// Memory Agent Model - Complete Pooling Approach
// (Single set of parameters shared by all agents)

data {
  int<lower = 1> trials;  // Number of trials per agent
  int<lower = 1> agents;  // Number of agents
  array[trials, agents] int h;  // Memory agent choices
  array[trials, agents] int other;  // Opponent (random agent) choices
}

parameters {
  // Single set of parameters shared by all agents
  real bias;  // Shared bias parameter
  real beta;  // Shared beta parameter
}

transformed parameters {
  // Memory state for each agent and trial
  array[trials, agents] real memory;
  
  // Calculate memory states
  for (agent in 1:agents){
    for (trial in 1:trials){
      if (trial == 1) {
        memory[trial, agent] = 0.5;
      } 
      if (trial < trials){
        memory[trial + 1, agent] = memory[trial, agent] + 
                                 ((other[trial, agent] - memory[trial, agent]) / trial);
        if (memory[trial + 1, agent] == 0){memory[trial + 1, agent] = 0.01;}
        if (memory[trial + 1, agent] == 1){memory[trial + 1, agent] = 0.99;}
      }
    }
  }
}

model {
  // Priors for shared parameters
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(beta | 0, 1);

  // Likelihood (same parameters for all agents)
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial, agent] | bias + memory[trial, agent] * beta);
    }
  }
}

generated quantities{
  // Single set of predictions for all agents
  int<lower=0, upper = trials> posterior_preds0;
  int<lower=0, upper = trials> posterior_preds1;
  int<lower=0, upper = trials> posterior_preds2;
  
  // Generate predictions
  posterior_preds0 = binomial_rng(trials, inv_logit(bias + 0 * beta));
  posterior_preds1 = binomial_rng(trials, inv_logit(bias + 1 * beta));
  posterior_preds2 = binomial_rng(trials, inv_logit(bias + 2 * beta));
}

