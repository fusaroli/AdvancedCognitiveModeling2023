
// Memory Agent Model - No Pooling Approach
// (Separate parameters for each agent, no sharing of information)

data {
  int<lower = 1> trials;  // Number of trials per agent
  int<lower = 1> agents;  // Number of agents
  array[trials, agents] int h;  // Memory agent choices
  array[trials, agents] int other;  // Opponent (random agent) choices
}

parameters {
  // Individual parameters for each agent (no population structure)
  array[agents] real bias;  // Individual bias parameters
  array[agents] real beta;  // Individual beta parameters
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
  // Separate priors for each agent (no pooling)
  for (agent in 1:agents) {
    target += normal_lpdf(bias[agent] | 0, 1);
    target += normal_lpdf(beta[agent] | 0, 1);
  }

  // Likelihood
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial, agent] | 
            bias[agent] + memory[trial, agent] * beta[agent]);
    }
  }
}

generated quantities{
  // Predictions with different memory values
  array[agents] int<lower=0, upper = trials> posterior_preds0;
  array[agents] int<lower=0, upper = trials> posterior_preds1;
  array[agents] int<lower=0, upper = trials> posterior_preds2;
  
  // Generate predictions
  for (agent in 1:agents){
    posterior_preds0[agent] = binomial_rng(trials, inv_logit(bias[agent] + 0 * beta[agent]));
    posterior_preds1[agent] = binomial_rng(trials, inv_logit(bias[agent] + 1 * beta[agent]));
    posterior_preds2[agent] = binomial_rng(trials, inv_logit(bias[agent] + 2 * beta[agent]));
  }
}

