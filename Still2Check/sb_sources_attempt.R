SimpleBayes_f <- function(bias, Source1, Source2){
  
  outcome <- inv_logit_scaled(bias + logit_scaled(Source1) + logit_scaled(Source2))
  
  return(outcome)
  
}

bias <- 0
trials <- seq(10)
Source1 <- seq(1,7, 1)
Source2 <- seq(1,4, 1)

db <- expand.grid(bias = bias, trials = trials, Source1 = Source1, Source2 = Source2)

for (n in seq(nrow(db))) {
  db$belief[n] <- SimpleBayes_f(db$bias[n], db$Source1[n]/8, db$Source2[n]/5)
  db$choice[n] <- rbinom(1,1, db$belief[n])
}

data_simpleBayes <- list(
  N = nrow(db),
  y = db$choice,
  Source1 = db$Source1,
  Source2 = db$Source2
)


stan_simpleBayesSources_model <- "

data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1;
  array[N] int Source2;
}

parameters {
  real bias;
  real w1;
  real w2;
  // Estimated rates
  array[N] real theta1;
  array[N] real theta2;
}

model {
  // Priors
  target +=  normal_lpdf(bias | 0, 1);
  target +=  normal_lpdf(w1 | 0, 1);
  target +=  normal_lpdf(w2 | 0, 1);
  // Estimating rates
  target +=  binomial_logit_lpmf(Source1 | 8, theta1); 
  target +=  binomial_logit_lpmf(Source2 | 4, theta2); 
  target +=  bernoulli_logit_lpmf(y | bias + w1 * to_vector(theta1) + w2 * to_vector(theta2));
  
}

"

write_stan_file(
  stan_simpleBayesSources_model,
  dir = "stan/",
  basename = "W9_SimpleBayesSources.stan")

file <- file.path("stan/W9_SimpleBayesSources.stan")
mod_simpleBayes <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE),
                                 stanc_options = list("O1"))
samplesSources_simple <- mod_simpleBayes$sample(
  data = data_simpleBayes,
  #fixed_param = TRUE,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  iter_warmup = 1500,
  iter_sampling = 3000,
  refresh = 500
)

data_temporalBayes <- list(
  N = nrow(db),
  y = db$choice,
  Source1 = db$Source1,
  marbles1 = rep(8, nrow(db))
)


stan_temporalBayesSources_model <- "

data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1; // how many red
  array[N] int marbles1; // how many sampled at that
}

parameters {
  real bias;
  real w1; 
  real w2;  
  array[N] real theta1; // Estimated rate
}

transformed parameters {
  array[N] real theta2;
  theta2[1] = 0;
  for (n in 2:N){
    theta2[n] = bias + w1 * theta1[n] + w2 * theta2[n-1];
    }
}

model {
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(w1 | 0, 1);
  target += normal_lpdf(w2 | 0, 1);
  target += binomial_logit_lpmf(Source1 | marbles1, theta1); 
  target += bernoulli_logit_lpmf(y | bias + w1 * to_vector(theta1) + w2 * to_vector(theta2));
}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real w1_prior;
  real w2_prior;
  bias_prior = normal_rng(0, 1) ;
  w1_prior = normal_rng(0, 1) ;
  w2_prior = normal_rng(0, 1) ;
  for (n in 1:N)
    log_lik[n]= bernoulli_logit_lpmf(y[n] | bias + w1 * theta1[n] + w2 * theta2[n]);
}


"

write_stan_file(
  stan_temporalBayesSources_model,
  dir = "stan/",
  basename = "W9_temporalBayesSources.stan")

file <- file.path("stan/W9_temporalBayesSources.stan")
mod_simpleBayes <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE),
                                 stanc_options = list("O1"))

samplesSources_simple <- mod_simpleBayes$sample(
  data = data_temporalBayes,
  #fixed_param = TRUE,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  iter_warmup = 1500,
  iter_sampling = 3000,
  refresh = 500
)


data_RLBayes <- list(
  N = nrow(db),
  y = db$choice,
  Source1 = db$Source1,
  Source2 = db$Source2,
  Truth = ifelse(db$Source1 > 4, 1, 0)
)


stan_RLBayesSources_model <- "

data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1;
  array[N] int Source2;
  array[N] int Truth;
}

parameters {
  real bias;
  real w1;
  real learningRate;
  // Estimated rates
  array[N] real theta1;
  array[N] real theta2;
}

model {
  array[N] real w2;
  // Priors
  target +=  normal_lpdf(bias | 0, 1);
  target +=  normal_lpdf(w1 | 0, 1);
  //target +=  normal_lpdf(w2[1] | 0, 1);
  // Estimating rates
  target +=  binomial_logit_lpmf(Source1 | 8, theta1); 
  target +=  binomial_logit_lpmf(Source2 | 4, theta2); 
  
  w2[1] = 1;
  for (n in 1:N){
    target +=  bernoulli_logit_lpmf(y[n] | bias + w1 * theta1[n] + w2[n] * theta2[n]);
    
    if (n < N){
      if (Truth[n] == Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (1 - w2[n]);  // Assuming max(w1) = 1
      }
      if (Truth[n] != Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (0 - w2[n]);  // Assuming max(w1) = 1
      }
    }
  }
  
}
"

write_stan_file(
  stan_RLBayesSources_model,
  dir = "stan/",
  basename = "W9_RLBayesSources.stan")

file <- file.path("stan/W9_RLBayesSources.stan")
mod_simpleBayes <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE),
                                 stanc_options = list("O1"))

samplesSources_simple <- mod_simpleBayes$sample(
  data = data_RLBayes,
  #fixed_param = TRUE,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  iter_warmup = 1500,
  iter_sampling = 3000,
  refresh = 500
)


###
newBlock = ifelse(lag(db$Source2) > 2, 1, 0)
newBlock[1] = 1
data_SamplingBayes <- list(
  N = nrow(db),
  ColorChoice = db$choice,
  Source1 = db$Source1,
  SamplingChoice = ifelse(db$Source2 > 2, 1, 0),
  newBlock = newBlock,
  marbles1 = rep(8, nrow(db))
)


stan_SamplingBayes_model <- "

data {
  int<lower=0> N;
  array[N] int ColorChoice;
  array[N] int SamplingChoice;
  array[N] int newBlock;
  array[N] int Source1; // how many red
  array[N] int marbles1; // how many sampled at that
}

parameters {
  real w1; 
  real w2;  
  array[N] real theta1; // Estimated rate
}

transformed parameters {
  array[N] real theta2;
  for (n in 1:N){
    if (newBlock[n] == 1) {
      theta2[n] = 0;
    } else {
      theta2[n] = w1 * theta1[n] + w2 * theta2[n-1];
    }
  }
}

model {
  target += normal_lpdf(w1 | 0, 1);
  target += normal_lpdf(w2 | 0, 1);
  target += binomial_logit_lpmf(Source1 | marbles1, theta1); 
  target += bernoulli_logit_lpmf(SamplingChoice | abs(w1 * to_vector(theta1) + w2 * to_vector(theta2)));
  for (n in 1:N){
    if (SamplingChoice[n] == 1){
      target += bernoulli_logit_lpmf(ColorChoice[n] | w1 * theta1[n] + w2 * theta2[n]);
    }
  }
}
"

write_stan_file(
  stan_SamplingBayes_model,
  dir = "stan/",
  basename = "W9_SamplingBayes.stan")

file <- file.path("stan/W9_SamplingBayes.stan")
mod_simpleBayes <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE),
                                 stanc_options = list("O1"))

samplesSources_simple <- mod_simpleBayes$sample(
  data = data_SamplingBayes,
  #fixed_param = TRUE,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  max_treedepth = 20,
  iter_warmup = 1500,
  iter_sampling = 3000,
  refresh = 500
)
