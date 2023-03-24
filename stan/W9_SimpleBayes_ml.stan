
data {
  int<lower=0> N;  // n of trials
  int<lower=0> S;  // n of participants
  array[N, S] int y;
  array[N, S] real<lower=0, upper = 1> Source1;
  array[N, S] real<lower=0, upper = 1> Source2;
}

transformed data{
  array[N, S] real l_Source1;
  array[N, S] real l_Source2;
  l_Source1 = logit(Source1);
  l_Source2 = logit(Source2);
}

parameters {
  real biasM;
  real biasSD;
  array[S] real z_bias;
}

transformed parameters {
  vector[S] biasC;
  vector[S] bias;
  biasC = biasSD * to_vector(z_bias);
  bias = biasM + biasC;
 }

model {
  target +=  normal_lpdf(biasM | 0, 1);
  target +=  normal_lpdf(biasSD | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  target += std_normal_lpdf(to_vector(z_bias));
  
  for (s in 1:S){
  target +=  bernoulli_logit_lpmf(y[,s] | bias[s] + 
                                          to_vector(l_Source1[,s]) + 
                                          to_vector(l_Source2[,s]));
  }
}




