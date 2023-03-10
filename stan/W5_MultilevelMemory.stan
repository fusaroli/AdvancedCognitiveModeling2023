
//
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
  array[agents] real bias;
  array[agents] real beta;
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
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(biasSD | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(betaSD | 0, .3)  -
    normal_lccdf(0 | 0, .3);

  target += normal_lpdf(bias | biasM, biasSD); 
  target += normal_lpdf(beta | betaM, betaSD); 
 
  for (agent in 1:agents)
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial,agent] | 
            bias[agent] +  logit(memory[trial, agent]) * (beta[agent]));
    }
}

generated quantities{
   real biasM_prior;
   real<lower=0> biasSD_prior;
   real betaM_prior;
   real<lower=0> betaSD_prior;
   
   real bias_prior;
   real beta_prior;
   
   int<lower=0, upper = trials> prior_preds0;
   int<lower=0, upper = trials> prior_preds1;
   int<lower=0, upper = trials> prior_preds2;
   int<lower=0, upper = trials> posterior_preds0;
   int<lower=0, upper = trials> posterior_preds1;
   int<lower=0, upper = trials> posterior_preds2;
   array[agents] int<lower=0, upper = trials> posterior_predsID0;
   array[agents] int<lower=0, upper = trials> posterior_predsID1;
   array[agents] int<lower=0, upper = trials> posterior_predsID2;
   
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   
   prior_preds0 = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
   prior_preds1 = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
   prior_preds2 = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
   posterior_preds0 = binomial_rng(trials, inv_logit(biasM + 0 * betaM));
   posterior_preds1 = binomial_rng(trials, inv_logit(biasM + 1 * betaM));
   posterior_preds2 = binomial_rng(trials, inv_logit(biasM + 2 * betaM));
    

  for (agent in 1:agents){
    posterior_predsID0[agent] = binomial_rng(trials, inv_logit(bias[agent] +  0 * beta[agent]));
	  posterior_predsID1[agent] = binomial_rng(trials, inv_logit(bias[agent] +  1 * beta[agent]));
	  posterior_predsID2[agent] = binomial_rng(trials, inv_logit(bias[agent] +  2 * beta[agent]));
    }
   
}


