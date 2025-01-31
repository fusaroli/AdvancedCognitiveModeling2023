
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> agents;
  int<lower=1> trials;
  array[trials, agents] int choice; // outcome - NB make prev choice from this
  array[trials, agents] int feedback; // input
  array[2] real initialValue; // expected value before starting
}

transformed data{
  array[trials, agents] dummy_pos;
  array[trials, agents] dummy_neg;
  
  for (agent in 1:agents){
    for (t in 1:trials){
      if (feedback[t,agent] == -1){
        dummy_pos[t,agent] = 0;
        dummy_neg[t,agent] = 1;
      }
      if (feedback[t,agent] == 1){
        dummy_pos[t,agent] = 1;
        dummy_neg[t,agent] = 0;
      }
    }
  }
  
}

// The parameters accepted by the model. 
parameters {
  real logit_learningRate_pos_mu;
  real logit_learningRate_neg_mu;
  real logInvTemperature_mu;
  
  vector<lower=0>[3] sigmas;
  matrix[3, agents] ID_z;
  
  cholesky_factor_corr[3] L_u;
}

transformed parameters {
  matrix[agents, 3] IDs;
  
  IDs = (diag_pre_multiply(sigmas, L_u) * ID_z);
  
}

// The model to be estimated.
model {
  
  array[2] real Value;
  vector[2] rate;
  vector[2] predError;
  
  target += normal_lpdf(logit_learningRate_pos_mu | 0, 1); //learningRate ~ beta(2,2);
  target += normal_lpdf(logit_learningRate_neg_mu | 0, 1); //learningRate ~ beta(2,2);
  target += normal_lpdf(logInvTemperature_mu | 0, 1); // logInvTemperature ~ normal(0,1);
  
  target += normal_lpdf(sigmas[1] | 0, .3) -
    normal_lccdf(0 | 0, .3); //learningRate ~ beta(2,2);
  target += normal_lpdf(sigmas[2] |  0, .3) -
    normal_lccdf(0 | 0, .3); //learningRate ~ beta(2,2);
  target += normal_lpdf(sigmas[3] |  0, .3) -
    normal_lccdf(0 | 0, .3); // logInvTemperature ~ normal(0,1);
  
  target += lkj_corr_cholesky_lpdf(L_u | 3);
  target += std_normal_lpdf(ID_z);
  
  
  for (agent in 1:agents){
  
  Value = initialValue;
  
  for (t in 1:trials){
    
    rate = softmax(exp(logInvTemperature_mu + IDs[agent, 3]) * to_vector(Value));
    target += categorical_lpmf(choice[t,agent] | rate);//choice[t] ~ categorical(rate);
    
    predError = feedback[t,agent] - to_vector(Value);
    Value = to_vector(Value) + inv_logit(logit_learningRate_neg_mu + IDs[agent, 1]) * predError * dummy_neg[t] + inv_logit(logit_learningRate_pos_mu + IDs[agent, 2]) * predError * dummy_pos[t] ;
    
  }
  }
}

generated quantities {
  array[trials] real log_lik;
  array[trials,2] real expectedValue_prior;
  array[trials,2] real expectedValue_posterior;
  
  real<lower = 0, upper = 1> learningRate_pos_prior;
  real<lower = 0, upper = 1> learningRate_neg_prior;
  real logInvTemperature_prior;
  real InvTemperature_prior;
  
  real rate_prior;
  real rate_posterior;
  
  real predError_prior;
  real predError_posterior;
  
  learningRate_pos_prior = beta_rng(2,2); //learningRate ~ beta(2,2);
  learningRate_neg_prior = beta_rng(2,2); //learningRate ~ beta(2,2);
  logInvTemperature_prior = normal_rng(0, 1); // logInvTemperature ~ normal(0,1);
  InvTemperature_prior = exp(logInvTemperature_prior);
  
  expectedValue_prior[1,] = initialValue;
  expectedValue_posterior[1,] = initialValue;
  
  for (t in 1:trials){
    
    rate_prior = softmax(invTemperature_prior * expectedValue_prior[t]);
    rate_posterior = softmax(invTemperature * expectedValue_posterior[t]);
    
    log_lik[t] = categorical_lpmf(choice[t] | rate_posterior);//choice[t] ~ categorical(rate);
    
    predError_prior = feedback[t] - expectedValue_prior[t,];
    predError_posterior = feedback[t] - expectedValue_posterior[t,];
    
    expectedValue_prior[t+1,] = expectedValue_prior[t,] + learningRate_neg * predError_prior * dummy_neg[t] + learningRate_pos * predError_prior * dummy_pos[t] ;
    expectedValue_posterior[t+1,] = expectedValue_posterior[t,] + learningRate_neg * predError_posterior * dummy_neg[t] + learningRate_pos * predError_prior * dummy_pos[t] ;
    
  }
  
}









