
// Generalized Context Model (GCM) - multilevel version

data {
    int<lower=1> nsubjects;  // number of subjects
    int<lower=1> ntrials;  // number of trials
    int<lower=1> nfeatures;  // number of predefined relevant features
    array[ntrials] int<lower=0, upper=1> cat_one; // true responses on a trial by trial basis
    array[ntrials, nsubjects] int<lower=0, upper=1> y;  // decisions on a trial by trial basis
    array[ntrials, nfeatures] real obs; // stimuli as vectors of features assuming all participants get the same sequence
    real<lower=0, upper=1> b;  // initial bias for category one over two

    // priors
    vector[nfeatures] w_prior_values;  // concentration parameters for dirichlet distribution <lower=1>
    array[2] real c_prior_values;  // mean and variance for logit-normal distribution
}

transformed data { // assuming all participants get the same sequence
    
    array[ntrials] int<lower=0, upper=1> cat_two; // dummy variable for category two over cat 1
    array[sum(cat_one)] int<lower=1, upper = ntrials> cat_one_idx; // array of which stimuli are cat 1
    array[ntrials - sum(cat_one)] int<lower = 1, upper = ntrials> cat_two_idx; //  array of which stimuli are cat 2
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
    real logit_c_M;    // Pop Mean of the scaling parameter (how fast similarity decrease with distance). 
    real<lower = 0> logit_c_SD;    // Pop SD of the scaling parameter (how fast similarity decrease with distance). 
    vector[nsubjects] logit_c;    // scaling parameter (how fast similarity decrease with distance). 
    
    simplex[nfeatures] weight;  // simplex means sum(w)=1
    real<lower=0> kappa;
    array[nsubjects] simplex[nfeatures] w_ind;    // weight parameter (how much attention should be paid to feature 1 related to feature 2 - summing up to 1)
}

transformed parameters {
    // parameter w
    vector[nfeatures] alpha = kappa * weight;
    
    // parameter c 
    vector<lower=0,upper=2>[nsubjects] c = inv_logit(logit_c)*2;  // times 2 as c is bounded between 0 and 2

    // parameter r (probability of response = category 1)
    array[ntrials, nsubjects] real<lower=0.0001, upper=0.9999> r;
    array[ntrials, nsubjects] real rr;

    for (sub in 1:nsubjects) {
      for (trial in 1:ntrials) {

        // calculate distance from obs to all exemplars
        array[(trial-1)] real exemplar_sim;
        for (e in 1:(trial-1)){
            array[nfeatures] real tmp_dist;
            for (feature in 1:nfeatures) {
                tmp_dist[feature] = w_ind[sub,feature]*abs(obs[e,feature] - obs[trial,feature]);
            }
            exemplar_sim[e] = exp(-c[sub] * sum(tmp_dist));
        }

        if (sum(cat_one[:(trial-1)])==0 || sum(cat_two[:(trial-1)])==0){  // if there are no examplars in one of the categories
            r[trial,sub] = 0.5;

        } else {
            // calculate similarity
            array[2] real similarities;
            
            array[sum(cat_one[:(trial-1)])] int tmp_idx_one = cat_one_idx[:sum(cat_one[:(trial-1)])];
            array[sum(cat_two[:(trial-1)])] int tmp_idx_two = cat_two_idx[:sum(cat_two[:(trial-1)])];
            similarities[1] = mean(exemplar_sim[tmp_idx_one]);
            similarities[2] = mean(exemplar_sim[tmp_idx_two]);

            // calculate r
            rr[trial,sub] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr[trial,sub] > 0.9999){
                r[trial,sub] = 0.9999;
            } else if (rr[trial,sub] < 0.0001) {
                r[trial,sub] = 0.0001;
            } else if (rr[trial,sub] > 0.0001 && rr[trial,sub] < 0.9999) {
                r[trial,sub] = rr[trial,sub];
            } else {
                r[trial,sub] = 0.5;
            }}}}}

model {
    // Priors
    target += exponential_lpdf(kappa | 0.1);
    target += dirichlet_lpdf(weight | w_prior_values);
    
    target += normal_lpdf(logit_c_M | c_prior_values[1], c_prior_values[2]);
    target += normal_lpdf(logit_c_SD | 0, 1) - normal_lccdf(0 | 0, 1);
    
    target += normal_lpdf(logit_c | logit_c_M, logit_c_SD);
    
    // Decision Data
    for (sub in 1:nsubjects){
      
      target += dirichlet_lpdf(w_ind[sub] | alpha);
      
      for (trial in 1:ntrials){
        target += bernoulli_lpmf(y[trial,sub] | r[trial,sub]);
      }
    }
}
