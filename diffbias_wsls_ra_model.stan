

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
  real loss_beta; // Beta parameter describing the participant's tendency to stick with the strategy given a loss. Like bias, this  parameter is also a probability bound between 0 and 1.
  real win_beta; // Beta parameter describing the participant's tendency to stick with the strategy given a win.
}

// Step 3: specify the model. In this case we are looking at a gaussian, with the parameters bias, loss_beta and win_beta, and priors on the bias, loss_beta and win_beta.
model {
//In the following three lines, we set the priors for our parameters. The priors for all of the parameters bias and the two betas are a gaussian distribution with a mean of 0 and an sd of 1 (uninformed priors).
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(loss_beta | 0, 1);
  target += normal_lpdf(win_beta | 0, 1);
  
  // xxx
  for (t in 1:n) {
    if (prevWon[t] == 0) {
      target += bernoulli_logit_lpmf(choice[t] | bias + loss_beta * F_c[t]);
    } else {
      target += bernoulli_logit_lpmf(choice[t] | bias + win_beta * F_c[t]);
    }
  }
}

