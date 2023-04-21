
// Generalized Context Model (GCM)

data {
    int<lower=1> ntrials;  // number of trials
    int<lower=1> nfeatures;  // number of predefined relevant features
    array[ntrials] int<lower=0, upper=1> cat_one; // true responses on a trial by trial basis
    array[ntrials] int<lower=0, upper=1> y;  // decisions on a trial by trial basis
    array[ntrials, nfeatures] real obs; // stimuli as vectors of features
    real<lower=0, upper=1> b;  // initial bias for category one over two

    // priors
    vector[nfeatures] w_prior_values;  // concentration parameters for dirichlet distribution <lower=1>
    array[2] real c_prior_values;  // mean and variance for logit-normal distribution
}

transformed data {
    array[ntrials] int<lower=0, upper=1> cat_two; // dummy variable for category two over cat 1
    array[sum(cat_one)] int<lower=1, upper=ntrials> cat_one_idx; // array of which stimuli are cat 1
    array[ntrials-sum(cat_one)] int<lower=1, upper=ntrials> cat_two_idx; //  array of which stimuli are cat 2
    int idx_one = 1; // Initializing 
    int idx_two = 1;
    for (i in 1:ntrials){
        cat_two[i] = abs(cat_one[i]-1);

        if (cat_one[i]==1){
            cat_one_idx[idx_one] = i;
            idx_one +=1;
        } else {
            cat_two_idx[idx_two] = i;
            idx_two += 1;
        }
    }
}

parameters {
    simplex[nfeatures] w;  // simplex means sum(w)=1
    real logit_c;
}

transformed parameters {
    // parameter c 
    real<lower=0, upper=2> c = inv_logit(logit_c)*2;  // times 2 as c is bounded between 0 and 2

    // parameter r (probability of response = category 1)
    array[ntrials] real<lower=0.0001, upper=0.9999> r;
    array[ntrials] real rr;

    for (i in 1:ntrials) {

        // calculate distance from obs to all exemplars
        array[(i-1)] real exemplar_sim;
        for (e in 1:(i-1)){
            array[nfeatures] real tmp_dist;
            for (j in 1:nfeatures) {
                tmp_dist[j] = w[j]*abs(obs[e,j] - obs[i,j]);
            }
            exemplar_sim[e] = exp(-c * sum(tmp_dist));
        }

        if (sum(cat_one[:(i-1)])==0 || sum(cat_two[:(i-1)])==0){  // if there are no examplars in one of the categories
            r[i] = 0.5;

        } else {
            // calculate similarity
            array[2] real similarities;
            
            array[sum(cat_one[:(i-1)])] int tmp_idx_one = cat_one_idx[:sum(cat_one[:(i-1)])];
            array[sum(cat_two[:(i-1)])] int tmp_idx_two = cat_two_idx[:sum(cat_two[:(i-1)])];
            similarities[1] = sum(exemplar_sim[tmp_idx_one]);
            similarities[2] = sum(exemplar_sim[tmp_idx_two]);

            // calculate r[i]
            rr[i] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr[i] > 0.9999){
                r[i] = 0.9999;
            } else if (rr[i] < 0.0001) {
                r[i] = 0.0001;
            } else if (rr[i] > 0.0001 && rr[i] < 0.9999) {
                r[i] = rr[i];
            } else {
                r[i] = 0.5;
            }
        }
    }
}

model {
    // Priors
    target += dirichlet_lpdf(w | w_prior_values);
    target += normal_lpdf(logit_c | c_prior_values[1], c_prior_values[2]);
    
    
    // Decision Data
    target += bernoulli_lpmf(y | r);
}

generated quantities {
    // priors
    simplex[nfeatures] w_prior = dirichlet_rng(w_prior_values);
    real logit_c_prior = normal_rng(c_prior_values[1], c_prior_values[2]);
    real<lower=0, upper=2> c_prior = inv_logit(logit_c_prior)*2;

    // prior pred
    array[ntrials] real<lower=0, upper=1> r_prior;
    array[ntrials] real rr_prior;
    for (i in 1:ntrials) {

        // calculate distance from obs to all exemplars
        array[(i-1)] real exemplar_dist;
        for (e in 1:(i-1)){
            array[nfeatures] real tmp_dist;
            for (j in 1:nfeatures) {
                tmp_dist[j] = w_prior[j]*abs(obs[e,j] - obs[i,j]);
            }
            exemplar_dist[e] = sum(tmp_dist);
        }

        if (sum(cat_one[:(i-1)])==0 || sum(cat_two[:(i-1)])==0){  // if there are no examplars in one of the categories
            r_prior[i] = 0.5;

        } else {
            // calculate similarity
            array[2] real similarities;
            
            array[sum(cat_one[:(i-1)])] int tmp_idx_one = cat_one_idx[:sum(cat_one[:(i-1)])];
            array[sum(cat_two[:(i-1)])] int tmp_idx_two = cat_two_idx[:sum(cat_two[:(i-1)])];
            similarities[1] = exp(-c_prior * sum(exemplar_dist[tmp_idx_one]));
            similarities[2] = exp(-c_prior * sum(exemplar_dist[tmp_idx_two]));

            // calculate r[i]
            rr_prior[i] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr_prior[i] == 1){
                r_prior[i] = 0.9999;
            } else if (rr_prior[i] == 0) {
                r_prior[i] = 0.0001;
            } else if (rr_prior[i] > 0 && rr_prior[i] < 1) {
                r_prior[i] = rr_prior[i];
            } else {
                r_prior[i] = 0.5;
            }
        }
    }

    array[ntrials] int<lower=0, upper=1> priorpred = bernoulli_rng(r_prior);


    // posterior pred
    array[ntrials] int<lower=0, upper=1> posteriorpred = bernoulli_rng(r);
    array[ntrials] int<lower=0, upper=1> posteriorcorrect;
    for (i in 1:ntrials) {
        if (posteriorpred[i] == cat_one[i]) {
            posteriorcorrect[i] = 1;
        } else {
            posteriorcorrect[i] = 0;
        }
    }
    
    
    // log likelihood
   array[ntrials] real log_lik;

   for (i in 1:ntrials) {
        log_lik[i] = bernoulli_lpmf(y[i] | r[i]);
   }

}
