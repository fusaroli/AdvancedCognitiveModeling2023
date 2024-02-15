
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
  real alphaM; // learning rate
  real temperatureM; // softmax inv.temp.
  vector<lower = 0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  matrix[agents,2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  target += normal_lpdf(alphaM | 0, 1);
  target += normal_lpdf(temperatureM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  target += std_normal_lpdf(to_vector(z_IDs));
  
  for (agent in 1:agents){
    value = initValue;
    
    for (t in 1:trials) {
      theta = softmax( inv_logit(temperatureM + IDs[agent,2]) * 20 * value); // action prob. computed via softmax
      target += categorical_lpmf(choice[t, agent] | theta);
      
      pe = feedback[t, agent] - value[choice[t, agent]]; // compute pe for chosen value only
      value[choice[t, agent]] = value[choice[t, agent]] + inv_logit(alphaM + IDs[agent,1]) * pe; // update chosen V
    }
  }
  
}


