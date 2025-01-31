
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower=1> trials;
  int<lower=1> agents;
  array[trials, agents] int<lower=1,upper=2> choice;
  array[trials, agents] int<lower=-1,upper=1> feedback;
} 

transformed data {
  vector[2] initValue;  // initial values for V
  initValue = rep_vector(0.0, 2);
}

parameters {
  real alphaPosM; // learning rate
  real alphaNegM; // learning rate
  real temperatureM; // softmax inv.temp.
  vector<lower = 0>[3] tau;
  matrix[3, agents] z_IDs;
  cholesky_factor_corr[3] L_u;
}

transformed parameters {
  matrix[agents,3] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  target += normal_lpdf(alphaPosM | 0, 1);
  target += normal_lpdf(alphaNegM | 0, 1);
  target += normal_lpdf(temperatureM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(tau[3] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  
  target += lkj_corr_cholesky_lpdf(L_u | 3);
  target += std_normal_lpdf(to_vector(z_IDs));
  
  for (agent in 1:agents){
    value = initValue;
    
    for (t in 1:trials) {
      theta = softmax( inv_logit(temperatureM + IDs[agent,3]) * 20 * value); // action prob. computed via softmax
      target += categorical_lpmf(choice[t, agent] | theta);
      
      pe = feedback[t, agent] - value[choice[t, agent]]; // compute pe for chosen value only
      if (pe < 0)
          value[choice[t, agent]] = value[choice[t, agent]] + inv_logit(alphaNegM + IDs[agent,1]) * pe; // update chosen V
        if (pe > 0)
          value[choice[t, agent]] = value[choice[t, agent]] + inv_logit(alphaPosM + IDs[agent,2]) * pe; // update chosen V
    }
  }
  
}

generated quantities{
  // Define priors
  real alphaPosM_prior; // learning rate
  real alphaNegM_prior; // learning rate
  real temperatureM_prior; // softmax inv.temp.
  real<lower = 0> tau_prior;
  real agent_alphaPos_prior;
  real agent_alphaNeg_prior;
  real agent_temperature_prior;
  
  alphaPosM_prior = normal_rng(0, 1);
  alphaNegM_prior = normal_rng(0, 1);
  temperatureM_prior = normal_rng(0, 1);
  tau_prior = normal_lb_rng(0, .3, 0);
  agent_alphaPos_prior = normal_rng(0, 1);
  agent_alphaNeg_prior = normal_rng(0, 1);
  agent_temperature_prior = normal_rng(0, 1);
  
  // Define predictive checks - choice & value
  array[trials] int predChoice_prior;
  array[trials,2] real expectedValue_prior;
  array[trials, agents] int predChoice_posterior;
  array[trials,agents, 2] real expectedValue_posterior;
  
  vector[2] theta_prior;
  vector[2] value;
  real pe;
  vector[2] theta_posterior;
  
  array[trials, agents] real log_lik;
  //
  
  // // Prior predictive checks
  value = initValue;
  for (t in 1:trials) {
      theta_prior = softmax(inv_logit(temperatureM_prior + agent_temperature_prior) * 20 * value); // action prob. computed via softmax
      predChoice_prior[t] = categorical_rng(theta_prior);
            
      pe = feedback[t, 1] - value[choice[t, 1]]; // compute pe for chosen value only
      if (pe < 0)
          expectedValue_prior[t, choice[t, 1]] = value[choice[t, 1]] + inv_logit(alphaNegM_prior + agent_alphaNeg_prior) * pe; // update chosen V
      if (pe > 0)
          expectedValue_prior[t, choice[t, 1]] = value[choice[t, 1]] + inv_logit(alphaPosM_prior + agent_alphaPos_prior) * pe; // update chosen V
  }
    
  // // Posterior predictive checks and log-likelihood
  value = initValue;
 for (agent in 1:agents){
    value = initValue;
    
    for (t in 1:trials) {
      theta_posterior = softmax( inv_logit(temperatureM + IDs[agent,2]) * 20 * value); // action prob. computed via softmax
      predChoice_posterior[t, agent] = categorical_rng(theta_posterior);
      log_lik[t,agent] = categorical_lpmf(choice[t, agent] | theta_posterior);
      pe = feedback[t, agent] - value[choice[t, agent]]; // compute pe for chosen value only
      if (pe < 0)
          expectedValue_posterior[t,agent, choice[t, agent]] = value[choice[t, agent]] + inv_logit(alphaNegM + IDs[agent,1]) * pe; // update chosen V
        if (pe > 0)
          expectedValue_posterior[t,agent, choice[t, agent]] = value[choice[t, agent]] + inv_logit(alphaPosM + IDs[agent,2]) * pe; // update chosen V
    }
  }   

  // Define log-likelihood for test
  
}

