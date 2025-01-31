# Pizza Stone Temperature Analysis
# This code models and predicts heating times for a pizza stone using a physics-based
# approach with Bayesian inference. The model accounts for different initial temperatures
# and flame temperatures, using appropriate log transformations.

# Load required libraries
library(tidyverse)
library(brms)
library(bayesplot)
library(tidybayes)
library(cmdstanr)

## First we need to explore the actual equation

# Function to calculate temperature
temp_calc <- function(t, Ti, Tinf, HOT) {
  Tinf + (Ti - Tinf) * exp(-HOT * t)
}

# Create dataframe with parameters
params <- expand_grid(
  time = seq(0, 1000, by = 10),
  Ti = c(-10, 0, 10, 20, 30),                    # Initial temps
  Tinf = c(200, 300, 400),               # Flame temps 
  HOT = c(0.001, 0.002, 0.004, 0.02)             # Heating coefficients
)

# Calculate temperatures
results <- params %>%
  mutate(temp = temp_calc(time, Ti, Tinf, HOT))

# Plot
ggplot(results, aes(x = time/60, y = temp, color = factor(Ti))) +
  geom_line() +
  facet_grid(
    Tinf ~ HOT, 
    labeller = labeller(
      HOT = function(x) paste("HOT =", x),
      Tinf = function(x) paste("Flame\nTemperature\n", x, "°C")
    )
  ) +
  labs(x = "Time (min)", 
       y = "Temperature (°C)",
       color = "Initial\nTemperature") +
  theme_bw()

# Read and prepare the data
# Note: Temperature is in Celsius, Time is in seconds

data <- tibble(
  Order = rep(0:18, 3),
  Seconds = rep(c(0, 175, 278, 333, 443, 568, 731, 773, 851, 912, 980, 
                  1040, 1074, 1124, 1175, 1237, 1298, 1359, 1394), 3),
  Temperature = c(15.1, 233, 244, 280, 289, 304, 343, NA, 333, 341, 320, 
                  370, 325, 362, 363, 357, 380, 376, 380,
                  14.5, 139.9, 153, 36.1, 254, 459, 263, 369, rep(NA, 11),
                  12.9, 149.5, 159, 179.4, 191.7, 201, 210, NA, 256, 257, 
                  281, 293, 297, 309, 318, 321, rep(NA, 3)),
  Rater = rep(c("N", "TR", "R"), each = 19)
)

# Normal model
fit_norm <- brm(
  Temperature ~ Seconds,
  data = data,
  family = gaussian,
  chains = 4,
  cores = 4
)

# Log-normal model
fit_lnorm <- brm(
  Temperature ~ Seconds,
  data = data,
  family = lognormal,
  chains = 4,
  cores = 4
)

# Compare models
loo_compare(loo(fit_norm), loo(fit_lnorm))


# Generate predictions
fitted_norm <- fitted(fit_norm, newdata = data) %>% 
  as_tibble() %>% 
  bind_cols(data)

fitted_lnorm <- fitted(fit_lnorm, newdata = data) %>%
  as_tibble() %>% 
  bind_cols(data)

# Plot
ggplot() +
  geom_ribbon(data = fitted_norm, 
              aes(x = Seconds/60, ymin = Q2.5, ymax = Q97.5), 
              alpha = 0.2, fill = "blue") +
  geom_ribbon(data = fitted_lnorm, 
              aes(x = Seconds/60, ymin = Q2.5, ymax = Q97.5), 
              alpha = 0.2, fill = "red") +
  geom_line(data = fitted_norm, aes(x = Seconds/60, y = Estimate, color = "Normal")) +
  geom_line(data = fitted_lnorm, aes(x = Seconds/60, y = Estimate, color = "Log-normal")) +
  geom_point(data = data, aes(x = Seconds/60, y = Temperature)) +
  labs(x = "Time (minutes)", y = "Temperature (°C)", color = "Model") +
  facet_wrap(~Rater, labeller = labeller(Rater = c(
    "N" = "Rater N", 
    "TR" = "Rater TR", 
    "R" = "Rater R"
  ))) +
  theme_bw()


# Prepare data for Stan
stan_data <- list(
  N = nrow(data %>% filter(!is.na(Temperature))),
  time = data %>% filter(!is.na(Temperature)) %>% pull(Seconds),
  temp = data %>% filter(!is.na(Temperature)) %>% pull(Temperature),
  n_raters = 3,
  rater = as.numeric(factor(data %>% 
                              filter(!is.na(Temperature)) %>% 
                              pull(Rater))),
  Ti = c(100,100, 100),
  Tinf = 450
)

# Stan model
stan_code <- "
data {
  int<lower=0> N;                   // Number of observations
  vector[N] time;                   // Time points
  vector[N] temp;                   // Observed temperatures
  int<lower=0> n_raters;           // Number of raters
  array[N] int<lower=1,upper=n_raters> rater;  // Rater indices
  vector[n_raters] Ti;             // Initial temperatures
  real Tinf;                       // Flame temperature
}

parameters {
  vector<lower=0>[n_raters] HOT;   // Heating coefficients
  vector<lower=0>[n_raters] sigma; // Measurement error
}

model {
  vector[N] mu;
  
  // Physics-based temperature prediction
  for (i in 1:N) {
    mu[i] = Tinf + (Ti[rater[i]] - Tinf) * exp(-HOT[rater[i]] * time[i]);
  }
  
  // Prior distributions
  target += normal_lpdf(HOT | 0.005, 0.005);    // Prior for heating rate
  target += exponential_lpdf(sigma | 1);         // Prior for measurement error
  
  // Likelihood
  target += normal_lpdf(temp | mu, sigma[rater]);
}
"

# Save model
writeLines(stan_code, "pizza_model.stan")

# Compile and fit
mod <- cmdstan_model("pizza_model.stan")
fit <- mod$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4
)

fit$summary()

#data <- data %>% mutate(
#  temp = Temperature,
#  Tinf = 900,
#  Ti = min(Temperature, na.rm=T),
#  time = Seconds
#)

#Hot_Formula <- bf(
#  temp ~ Tinf + (Ti - Tinf) * exp(-HOT * time), 
#  HOT ~ 1 + (1|Rater), 
#  nl = TRUE )

#Hot_prior <- c(
#  prior(normal(0.005, 0.005), nlpar = "HOT", class = "b",  lb = 0),
#  prior(exponential(1), class = "sigma"))

#Hot_Model <- brm(
#  Hot_Formula, 
#  prior = Hot_prior,
#  data = data, 
#  backend = "cmdstanr",
#  chains = 4, 
#  cores = 4, 
#  control = list(adapt_delta = 0.99))


# Extract draws
post <- as_draws_df(fit$draws()) %>%
  select(starts_with("HOT"), starts_with("sigma")) %>%
  slice_sample(n = 100)

# Prediction grid
pred_data <- crossing(
  time = seq(0, max(stan_data$time), length.out = 100),
  rater = 1:stan_data$n_raters
) %>%
  mutate(
    Ti = stan_data$Ti[rater],
    Tinf = stan_data$Tinf
  )

# Generate predictions
pred_matrix <- matrix(NA, nrow = nrow(pred_data), ncol = 100)
for (i in 1:nrow(pred_data)) {
  pred_matrix[i,] <- with(pred_data[i,], 
                          Tinf + (Ti - Tinf) * exp(-as.matrix(post)[,rater] * time)
  )
}

# Summarize predictions
predictions <- pred_data %>%
  mutate(
    mean = rowMeans(pred_matrix),
    lower = apply(pred_matrix, 1, quantile, 0.025),
    upper = apply(pred_matrix, 1, quantile, 0.975)
  )

# Plot
ggplot(predictions, aes(x = time/60)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  geom_line(aes(y = mean)) +
  geom_point(data = data %>% 
               filter(!is.na(Temperature)) %>%
               mutate(rater = case_when(
                 Rater == "N" ~ 1,
                 Rater == "TR" ~ 2,
                 Rater == "R" ~ 3
               )), 
             aes(x = Seconds/60, y = Temperature)) +
  facet_wrap(~rater, labeller = labeller(rater = c(
    "1" = "Rater N", 
    "2" = "Rater TR", 
    "3" = "Rater R"
  ))) +
  labs(x = "Time (minutes)", y = "Temperature (°C)") +
  theme_bw()


time_to_temp <- function(target_temp, HOT, Ti, Tinf) {
  # Solve: target = Tinf + (Ti - Tinf) * exp(-HOT * t)
  # for t
  t = -1/HOT * log((target_temp - Tinf)/(Ti - Tinf))
  return(t/60)  # Convert seconds to minutes
}

# Extract HOT samples
hot_samples <- as_draws_df(fit$draws()) %>%
  select(starts_with("HOT"))

# Create grid of flame temperatures
pred_data <- crossing(
  Tinf = seq(450, 1200, by = 50),
  rater = 1:3
) %>%
  mutate(
    Ti = stan_data$Ti[rater],
    target_temp = 400
  )

# Calculate times for each HOT posterior sample
n_samples <- 100
# Create time predictions
time_preds <- map_dfr(1:nrow(pred_data), function(i) {
  times <- sapply(1:n_samples, function(j) {
    hot <- hot_samples[j, paste0("HOT[", pred_data$rater[i], "]")][[1]]
    time_to_temp(pred_data$target_temp[i], hot, pred_data$Ti[i], pred_data$Tinf[i])
  })
  
  data.frame(
    rater = pred_data$rater[i],
    Tinf = pred_data$Tinf[i],
    mean_time = mean(times),
    lower = quantile(times, 0.025),
    upper = quantile(times, 0.975)
  )
})

# Plot
ggplot(time_preds, aes(x = Tinf)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  geom_line(aes(y = mean_time)) +
  facet_wrap(~rater, labeller = labeller(rater = c(
    "1" = "Rater N", 
    "2" = "Rater TR", 
    "3" = "Rater R"
  ))) +
  labs(x = "Flame Temperature (°C)", 
       y = "Minutes to reach 400°C") +
  theme_bw()
