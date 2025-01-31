
data {
  //trial number
  int<lower=0> N;
  //binary responses
  array[N] int resp;
  //contingency
  vector[N] u;
  
}

parameters {
  
  //learning rate
  real omega;
  // inital expectation
  real E_i;
  //inital sigma2
  real sa2_i;
  
  
}

transformed parameters{
  //expectations
  vector[N+1] mu1hat;
  
  vector[N+1] pe1;
  
  vector[N+1] sa1hat;
  
  vector[N+1] mu2;
  
  vector[N+1] sa2;
  
  vector[N+1] sa2hat;
    
  
  mu2[1] = inv_logit(E_i);
  
  sa2[1] = exp(sa2_i);
  
  // sa2[1] = 4;
  
  //updating expectation
  for(i in 1:N){
  
    mu1hat[i] = inv_logit(mu2[i]);   //equation (24)
    
    // prediction error for trial.
    pe1[i] = u[i]-mu1hat[i];
    
    //updating beliefs based on prediction error
    //equation 23
    mu2[i+1] = mu2[i]+sa2[i]*pe1[i];
    
    //equation(26)
    sa1hat[i+1] = mu1hat[i] * (1 - mu1hat[i]); 
  
    // equation (27) when mu3 = 0
    sa2hat[i+1] = sa2[i] + exp(omega); 
    
    //eqaution (22)
    sa2[i+1] = 1 / ((1/sa2hat[i+1]) + sa1hat[i+1]);
    
  }
  
  
  
}

model {
  //priors
  
  //omega //learning rate
  omega ~ normal(0,10);
  // inital expectation
  E_i ~ normal(0,1);
  //inital sigma2
  sa2_i ~ normal(0,2);
 
  // we have expectations for the next trial if the experiment kept on going i.e. N+1 but we only have responses for 1:N
  resp ~ bernoulli(mu1hat[1:N]);

}

