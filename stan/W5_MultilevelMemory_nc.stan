

//
// This STAN model is a multilevel memory agent
//
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input (data) for the model. 
data {
 int<lower = 1> trials;
 int<lower = 1> agents;
 array[trials, agents] int h;
 array[trials, agents] int other;
}

// The parameters accepted by the model. 
parameters {
  real biasM;
  real<lower = 0> biasSD;
  real betaM;
  real<lower = 0> betaSD;
  vector[agents] biasID_z;
  vector[agents] betaID_z;
}

transformed parameters {
  array[trials, agents] real memory;
  vector[agents] biasID;
  vector[agents] betaID;
  
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
  biasID = biasID_z * biasSD;
  betaID = betaID_z * betaSD;
 }

// The model to be estimated. 
model {
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(biasSD | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(betaSD | 0, .3)  -
    normal_lccdf(0 | 0, .3);

  target += std_normal_lpdf(to_vector(biasID_z)); // target += normal_lpdf(to_vector(biasID_z) | 0, 1);
  target += std_normal_lpdf(to_vector(betaID_z)); // target += normal_lpdf(to_vector(betaID_z) | 0, 1);
 
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial,agent] | 
            biasM + biasID[agent] +  logit(memory[trial, agent]) * (betaM + betaID[agent]));
    }
  }
  
  
}

generated quantities{
   real biasM_prior;
   real<lower=0> biasSD_prior;
   real betaM_prior;
   real<lower=0> betaSD_prior;
   
   real bias_prior;
   real beta_prior;
   
   array[agents] int<lower=0, upper = trials> prior_preds0;
   array[agents] int<lower=0, upper = trials> prior_preds1;
   array[agents] int<lower=0, upper = trials> prior_preds2;
   array[agents] int<lower=0, upper = trials> posterior_preds0;
   array[agents] int<lower=0, upper = trials> posterior_preds1;
   array[agents] int<lower=0, upper = trials> posterior_preds2;
   
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   


for (agent in 1:agents){
    prior_preds0[agent] = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
    prior_preds1[agent] = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
    prior_preds2[agent] = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
	  posterior_preds0[agent] = binomial_rng(trials, inv_logit(biasM + biasID[agent] +  0 * (betaM + betaID[agent])));
	  posterior_preds1[agent] = binomial_rng(trials, inv_logit(biasM + biasID[agent] +  1 * (betaM + betaID[agent])));
	  posterior_preds2[agent] = binomial_rng(trials, inv_logit(biasM + biasID[agent] +  2 * (betaM + betaID[agent])));
    }
}


