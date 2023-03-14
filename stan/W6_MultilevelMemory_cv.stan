
//
// This STAN model infers a random bias from a sequences of 1s and 0s (heads and tails)
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
 
 int<lower = 1> agents_test;
 array[trials, agents_test] int h_test;
 array[trials, agents_test] int other_test;
}

// The parameters accepted by the model. 
parameters {
  real biasM;
  real betaM;
  vector<lower = 0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  array[trials, agents] real memory;
  array[trials, agents_test] real memory_test;
  matrix[agents,2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
  
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
  
  for (agent in 1:agents_test){
    for (trial in 1:trials){
      if (trial == 1) {
        memory_test[trial, agent] = 0.5;
      } 
      if (trial < trials){
        memory_test[trial + 1, agent] = memory_test[trial, agent] + ((other[trial, agent] - memory[trial, agent]) / trial);
        if (memory_test[trial + 1, agent] == 0){memory_test[trial + 1, agent] = 0.01;}
        if (memory_test[trial + 1, agent] == 1){memory_test[trial + 1, agent] = 0.99;}
      }
    }
  }
  
}

// The model to be estimated. 
model {
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += lkj_corr_cholesky_lpdf(L_u | 2);

  target += std_normal_lpdf(to_vector(z_IDs));
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial, agent] | biasM + IDs[agent, 1] +  memory[trial, agent] * (betaM + IDs[agent, 2]));
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
   
   array[trials, agents] real log_lik;
   array[trials, agents_test] real log_lik_test;
   
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   
   for (i in 1:agents){
      prior_preds0[i] = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
      prior_preds1[i] = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
      prior_preds2[i] = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
      posterior_preds0[i] = binomial_rng(trials, inv_logit(biasM + IDs[i,1] +  0 * (betaM + IDs[i,2])));
      posterior_preds1[i] = binomial_rng(trials, inv_logit(biasM + IDs[i,1] +  1 * (betaM + IDs[i,2])));
      posterior_preds2[i] = binomial_rng(trials, inv_logit(biasM + IDs[i,1] +  2 * (betaM + IDs[i,2])));
      
      for (t in 1:trials){
        log_lik[t,i] = bernoulli_logit_lpmf(h[t, i] | biasM + IDs[i, 1] +  memory[t, i] * (betaM + IDs[i, 2]));
      }
   }
   
   for (i in 1:agents_test){
    for (t in 1:trials){
      log_lik_test[t,i] = bernoulli_logit_lpmf(h_test[t,i] | biasM +  memory_test[t, i] * betaM);
    }
  }
  
}

