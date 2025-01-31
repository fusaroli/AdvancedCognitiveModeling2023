
data {
    int<lower=1> trials;
    array[trials] int<lower=1,upper=2> choice;
    array[trials] int<lower=-1,upper=1> feedback;
} 

transformed data {
  vector[2] initValue;  // initial values for V
  initValue = rep_vector(0.0, 2);
}

parameters {
    real<lower=0, upper=1> alpha; // learning rate
    real<lower=0, upper=20> temperature; // softmax inv.temp.
}

model {
    real pe;
    vector[2] value;
    vector[2] theta;
    
    target += uniform_lpdf(alpha | 0, 1);
    target += uniform_lpdf(temperature | 0, 20);
    
    value = initValue;
    
    for (t in 1:trials) {
        theta = softmax( temperature * value); // action prob. computed via softmax
        target += categorical_lpmf(choice[t] | theta);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
    
}

generated quantities{
  real<lower=0, upper=1> alpha_prior;
  real<lower=0, upper=20> temperature_prior;
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  real log_lik;
  
  alpha_prior = uniform_rng(0,1);
  temperature_prior = uniform_rng(0,20);
  
  value = initValue;
  log_lik = 0;
  
  for (t in 1:trials) {
        theta = softmax( temperature * value); // action prob. computed via softmax
        log_lik = log_lik + categorical_lpmf(choice[t] | theta);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
  
}

