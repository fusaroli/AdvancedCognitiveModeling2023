

// Step 1: specify the input
data {
 int<lower=1> n;
 array[n] int choice;
 array[n] int F_c;
 array[n] int prevWon; // Whether the participant won or lost the previous round. It is specified as an integer and is coded 1 or won and -1 for lost.
}

// Step 2: specify the parameters
parameters {
  real bias;
  real loss_beta; // Beta parameter describing the participant's tendency to stick with the strategy given a loss.
  real win_beta; // Beta parameter describing the participant's tendency to stick with the strategy given a win.
}

// Step 3: specify the model. In this case we are looking at a gaussian, with the parameters bias, loss_beta and win_beta, and priors on the bias, loss_beta and win_beta.
model {
//In the following three lines, we set the priors for our parameters. The priors for all of the parameters bias and the two betas are a gaussian distribution with a mean of 0 and an sd of 1 (uninformed priors).
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(loss_beta | 0, 1);
  target += normal_lpdf(win_beta | 0, 1);
  
  // For the model, we make a for loop over the trials, and in the loop we have an if/else statement for whether the agent won or lost on the previous trial. The model is similar to our first model, except if the agent lost on the previous trial, then it uses the loss-specific beta value, and if it won the it uses the winning-specific beta-value.
  for (t in 1:n) {
    if (prevWon[t] == 0) {
      target += bernoulli_logit_lpmf(choice[t] | bias + loss_beta * to_vector(F_c)[t]);
    } else {
      target += bernoulli_logit_lpmf(choice[t] | bias + win_beta * to_vector(F_c)[t]);
    }
  }
}

