
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
    real<lower=0, upper=1> alpha_pos; // learning rate
    real<lower=0, upper=1> alpha_neg; // learning rate
    real<lower=0, upper=20> temperature; // softmax inv.temp.
}

model {
    real pe;
    vector[2] value;
    vector[2] theta;
    
    target += uniform_lpdf(alpha_pos | 0, 1);
    target += uniform_lpdf(alpha_neg | 0, 1);
    target += uniform_lpdf(temperature | 0, 20);
    
    value = initValue;
    
    for (t in 1:trials) {
        theta = softmax( temperature * value); // action prob. computed via softmax
        target += categorical_lpmf(choice[t] | theta);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        
        if (pe < 0)
          value[choice[t]] = value[choice[t]] + alpha_neg * pe; // update chosen V
        if (pe > 0)
          value[choice[t]] = value[choice[t]] + alpha_pos * pe; // update chosen V
    }
    
}

