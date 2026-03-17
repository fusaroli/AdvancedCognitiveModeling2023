
data {
  int<lower=1> N_train; 
  int<lower=1> J_train;
  array[N_train] int<lower=1, upper=J_train> agent_train;
  array[N_train] int<lower=0, upper=1> h_train;
  vector<lower=0.01, upper=0.99>[N_train] opp_rate_prev_train;

  int<lower=1> N_test; 
  int<lower=1> J_test;
  array[N_test] int<lower=1, upper=J_test> agent_test;
  array[N_test] int<lower=0, upper=1> h_test;
  vector<lower=0.01, upper=0.99>[N_test] opp_rate_prev_test;
}

parameters {
  real mu_alpha; 
  real mu_beta; 
  vector<lower=0>[2] sigma;
  cholesky_factor_corr[2] L_Omega; 
  matrix[2, J_train] z;
}

transformed parameters {
  matrix[J_train, 2] indiv_params;
  {
    matrix[2, J_train] dev = diag_pre_multiply(sigma, L_Omega) * z;
    for (j in 1:J_train) {
      indiv_params[j, 1] = mu_alpha + dev[1, j];
      indiv_params[j, 2] = mu_beta + dev[2, j];
    }
  }
}

model {
  target += normal_lpdf(mu_alpha | 0, 1); 
  target += normal_lpdf(mu_beta | 0, 1);
  target += exponential_lpdf(sigma | 1);
  target += lkj_corr_cholesky_lpdf(L_Omega | 2);
  target += std_normal_lpdf(to_vector(z));
  
  vector[N_train] logit_p;
  for (i in 1:N_train) {
    logit_p[i] = indiv_params[agent_train[i], 1] + 
                 indiv_params[agent_train[i], 2] * logit(opp_rate_prev_train[i]);
  }
  target += bernoulli_logit_lpmf(h_train | logit_p);
}

generated quantities {
  matrix[2, J_test] new_indiv;
  vector[N_test] log_lik_test;
  
  for (j in 1:J_test) {
    vector[2] z_new;
    z_new[1] = std_normal_rng();
    z_new[2] = std_normal_rng();
    vector[2] param_new = [mu_alpha, mu_beta]' + diag_pre_multiply(sigma, L_Omega) * z_new;
    new_indiv[1, j] = param_new[1];
    new_indiv[2, j] = param_new[2];
  }
  
  for (n in 1:N_test) {
    real lp = new_indiv[1, agent_test[n]] + 
              new_indiv[2, agent_test[n]] * logit(opp_rate_prev_test[n]);
    log_lik_test[n] = bernoulli_logit_lpmf(h_test[n] | lp);
  }
}
