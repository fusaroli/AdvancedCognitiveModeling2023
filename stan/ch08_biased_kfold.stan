
data {
  int<lower=1> N_train; 
  int<lower=1> J_train;
  array[N_train] int<lower=1, upper=J_train> agent_train;
  array[N_train] int<lower=0, upper=1> h_train;
  
  int<lower=1> N_test; 
  int<lower=1> J_test;
  array[N_test] int<lower=1, upper=J_test> agent_test;
  array[N_test] int<lower=0, upper=1> h_test;
}

parameters {
  real mu_theta; 
  real<lower=0> sigma_theta; 
  vector[J_train] z_theta;
}

transformed parameters {
  vector[J_train] theta_logit = mu_theta + z_theta * sigma_theta;
}

model {
  target += normal_lpdf(mu_theta | 0, 1.5);
  target += exponential_lpdf(sigma_theta | 1);
  target += std_normal_lpdf(z_theta);
  target += bernoulli_logit_lpmf(h_train | theta_logit[agent_train]);
}

generated quantities {
  vector[J_test] theta_new;
  vector[N_test] log_lik_test;
  
  for (j in 1:J_test) {
    theta_new[j] = normal_rng(mu_theta, sigma_theta);
  }
  
  for (n in 1:N_test) {
    log_lik_test[n] = bernoulli_logit_lpmf(h_test[n] | theta_new[agent_test[n]]);
  }
}
