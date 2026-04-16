
functions {
  real jardri_f(real L, real w) {
    real num = w * exp(L) + (1.0 - w);
    real den = (1.0 - w) * exp(L) + w;
    return log(num / den);
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
  array[N] real lo_self;
  array[N] real lo_social;
  
  for (i in 1:N) {
    lo_self[i]   = log((alpha0 + blue1[i]) / (beta0 + total1[i] - blue1[i]));
    lo_social[i] = log((alpha0 + blue2[i]) / (beta0 + total2[i] - blue2[i]));
  }
}

parameters {
  real<lower=0.5, upper=1.0> w_self;
  real<lower=0.5, upper=1.0> w_other;
  real<lower=0> alpha_self_m1;    // Excess over 1
  real<lower=0> alpha_other_m1;
}

transformed parameters {
  // Reconstruct the actual loop multiplier
  real<lower=1> alpha_self  = 1.0 + alpha_self_m1;
  real<lower=1> alpha_other = 1.0 + alpha_other_m1;
}

model {
  w_self  ~ normal(0.75, 0.25);
  w_other ~ normal(0.75, 0.25);
  // Half-normal priors shrinking towards 1 (no overcounting)
  alpha_self_m1  ~ std_normal();
  alpha_other_m1 ~ std_normal();
  
  for (i in 1:N) {
    // Evidence is multiplied by the loop parameter BEFORE weighting
    real L_post = jardri_f(alpha_self * lo_self[i], w_self) + 
                  jardri_f(alpha_other * lo_social[i], w_other);
                  
    target += bernoulli_logit_lpmf(choice[i] | L_post);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;
  
  for (i in 1:N) {
    real L_post = jardri_f(alpha_self * lo_self[i], w_self) + 
                  jardri_f(alpha_other * lo_social[i], w_other);
                  
    log_lik[i]        = bernoulli_logit_lpmf(choice[i] | L_post);
    posterior_pred[i] = bernoulli_logit_rng(L_post);
  }
}

