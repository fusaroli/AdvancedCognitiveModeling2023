
data {
  int<lower=0> N;  // n of trials
  int<lower=0> S;  // n of participants
  array[N, S] int y;
  array[N, S] real<lower=0, upper = 1> Source1;
  array[N, S] real<lower=0, upper = 1> Source2; 
}

transformed data {
  array[N,S] real l_Source1;
  array[N,S] real l_Source2;
  l_Source1 = logit(Source1);
  l_Source2 = logit(Source2);
}

parameters {
  real biasM;
  
  real w1_M; 
  real w2_M;
  
  vector<lower = 0>[3] tau;
  matrix[3, S] z_IDs;
  cholesky_factor_corr[3] L_u;
  
}

transformed parameters{
  matrix[S,3] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {

  target += normal_lpdf(biasM | 0, 1);
  target +=  normal_lpdf(tau[1] | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  target += normal_lpdf(w1_M | 0, 1);
  target +=  normal_lpdf(tau[2] | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  target += normal_lpdf(w2_M | 0, 1);
  target +=  normal_lpdf(tau[3] | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  target += lkj_corr_cholesky_lpdf(L_u | 3);
  
  target += std_normal_lpdf(to_vector(z_IDs));
    
  for (s in 1:S){
  for (n in 1:N){
    target += bernoulli_logit_lpmf(y[n,s] | biasM + IDs[s, 1] + 
          (w1_M + IDs[s, 2]) * l_Source1[n,s] + 
          (w2_M + IDs[s, 3]) * l_Source2[n,s]);
  }}
}


