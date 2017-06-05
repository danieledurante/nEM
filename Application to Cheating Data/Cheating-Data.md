Performance assessment for the CHEATING data
================
Daniele Durante

Description
-----------

This tutorial implementation focuses on assessing the maximization performance and the computational efficiency of the different algorithms for the estimation of latent class models with covariates. In particular, this assessment considers the dataset `cheating` from the `R` library [`poLCA`](https://www.jstatsoft.org/article/view/v042i10).

The analyses reproduce those discussed in Section 3.1 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), where we propose a novel **nested EM** algorithm for improved maximum likelihood estimation of latent class models with covariates.

Load the source functions and the data
--------------------------------------

The functions for the implementation of the different algorithms (including the **nested EM**, and the popular competitors currently considered in the literature) are available in the source file [`LCA-Covariates-Algorithms.R`](https://github.com/danieledurante/nEM/blob/master/LCA-Covariates-Algorithms.R). More comments on the different maximization routines can be found in the file `LCA-Covariates-Algorithms.R`.

Let us load this source file, along with the `cheating` dataset.

``` r
rm(list = ls())
source("LCA-Covariates-Algorithms.R")
data(cheating)
str(cheating)
```

    ## 'data.frame':    319 obs. of  5 variables:
    ##  $ LIEEXAM : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LIEPAPER: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ FRAUD   : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ COPYEXAM: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ GPA     : int  NA NA NA NA 1 1 1 1 1 1 ...

Since the main focus is on comparing the computational performance of the different algorithms, let us for simplicity remove all the statistical units having missing values.

``` r
cheating <- na.omit(cheating)
```

Consistent with the analyses in Section 3.1 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), we focus on the latent class model for the `J = 4` different cheating behaviors, having the variable `GPA` (grade point average) as a covariate in the multinomial logistic regression for the latent classes. Using the syntax of the library `poLCA`, this model can be defined as follows:

``` r
f_cheating <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
```

To provide a detailed computational assessment, we perform estimation under the different algorithms for `Rep_Tot = 100` runs at varying initializations. For each run, the algorithms are all initialized at the same starting values, which are controlled by a `seed` (changing across the different runs). Let us, therefore, define this 100 seeds.

``` r
Rep_Tot <- 100
seed_rep <- c(101:200)
seed_rep[35] <- 1
seed_rep[89] <- 2
```

Note that in the above `seed_rep` specification, some values are tuned since the one-step EM algorithm incorporating Newton-Raphson methods converged to undefined log-likelihoods in some runs. Hence, we changed some seeds to improve the behavior of this competing algorithm.

Estimation under the different maximization routines
----------------------------------------------------

We perform estimation of the parameters in the above latent class model with covariates under different computational routines (including our novel **nested EM** algorithm), and compare maximization performance along with computational efficiency.

Consistent with the tutorial analyses in [Linzer and Lewis (2011)](https://www.jstatsoft.org/article/view/v042i10), we focus on the model with `R = 2` latent classes.

#### 1. EM algorithm with Newton-Raphson methods (one-step maximization)

Here we consider the one-step EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407), and discussed in Section 1.1 of our paper. This requires the function `newton_em()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before.

Let us first create the quantities to be monitored for each run. These include the number of iterations to reach convergence, the log-likelihood sequence, and a vector monitoring presence (1) or absence (0) of drops in the log-likelihood sequence at each iteration.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_1 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_1 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_1 <- matrix(0, Rep_Tot, 1000)

# 1000 means that the maximum number of iteration of the EM we will consider is 1000. 
```

Finally, let us perform the `Rep_Tot = 100` runs of the one-step EM algorithm with Newton-Raphson methods. We also monitor the computational time via the function `system.time()`.

``` r
# Perform the algorithm.
time_NR_EM_alpha_1 <- system.time(
  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep])
iter_NR_EM_alpha_1[rep] <- fit_NR_EM[[1]]   
llik_NR_EM_alpha_1[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_1[rep,] <- fit_NR_EM[[3]]}

)[3]
```

#### 2. Re-scaled EM algorithm with Newton-Raphson methods (one-step maximization)

Here we consider the re-scaled version of the above one-step EM algorithm with Newton-Raphson methods. This modification is discussed in Section 1.1 of our paper, and its general version can be found in Chapter 1.5.6 of [McLachlan and Krishnan (2007)](http://onlinelibrary.wiley.com/book/10.1002/9780470191613). Also this algorithm requires the function `newton_em()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before. However, now the parameter 0 &lt; *α* &lt; 1 should be modified to reduce concerns about drops in the log-likelihood sequence. Here we consider:

-   The case *α* = 0.75.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.75 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.75 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.75 <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NR_EM_alpha_0.75 <- system.time(   
  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.75)
iter_NR_EM_alpha_0.75[rep] <- fit_NR_EM[[1]]    
llik_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[3]]}

)[3]
```

-   The case *α* = 0.50.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.5 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.5 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.5 <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NR_EM_alpha_0.5 <- system.time(    
  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.5)
iter_NR_EM_alpha_0.5[rep] <- fit_NR_EM[[1]] 
llik_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[3]]}

)[3]
```

-   The case *α* = 0.25.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.25 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.25 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.25 <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NR_EM_alpha_0.25 <- system.time(   
  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.25)
iter_NR_EM_alpha_0.25[rep] <- fit_NR_EM[[1]]    
llik_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[3]]}

)[3]
```

#### 3. Classical 3-step algorithm (three-step maximization)

Here we consider the classical three-step strategy to estimate latent class models with covariates (e.g. [Clogg 1995](https://www.iser.essex.ac.uk/research/publications/494549)). As discussed in Section 1.2 of our paper, this algorithm consists of three steps.

1.  Estimate a latent class model without covariates `f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1`. This requires the function `unconditional_em()` (in `LCA-Covariates-Algorithms.R`).
2.  Using the estimates in 1, predict the latent classes *s*<sub>*i*</sub>, *i* = 1, ..., *n*, by assigning each unit *i* to the class *r* with the highest predicted probability.
3.  Using the `R` function `multinom` in the library `nnet`, estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a multinomial logistic regression with the predicted *s*<sub>1</sub>, ..., *s*<sub>*n*</sub> as responses.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_classical <- rep(0, Rep_Tot)

# Define useful quantities to compute the full--model log-likelihood sequence.
f_cheating_3_step <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
nclass = 2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

# Perform the three step algorithm.
time_3_step_classical <- system.time(
  
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
fit_unconditional <- unconditional_em(f_cheating_unconditional, 
                                      cheating, nclass = 2, seed = seed_rep[rep])

#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]],1, which.max)

#---------------------------------------------------------------------------------------------------
# 3] Estimate the beta coefficient from a multinomial logit with predicted classes as responses
#---------------------------------------------------------------------------------------------------
b <- c(t(summary(multinom(pred_class ~ cheating$GPA, trace = FALSE))$coefficients))

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(b, x_cheating_3_step, R_cheating_3_step)
llik_3_step_classical[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],
                                  y_cheating_3_step))))}

)[3]
```

#### 4. Bias-corrected 3-step algorithm (three-step maximization)

Here we implement the modification proposed by [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) of the classical three-step methods, in order to reduce the bias of the estimators. This strategy is discussed in Sections 1.2 and 4 of our paper, and proceed as follows:

1.  Estimate a latent class model without covariates `f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1`. This requires the function `unconditional_em()` (in `LCA-Covariates-Algorithms.R)`.
2.  Using the estimates in 1, predict the latent classes *s*<sub>*i*</sub>, *i* = 1, ..., *n*, by assigning each unit *i* to the class *r* with the highest predicted probability. Compute also the classification error by applying equation (6) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved).
3.  Following equation (19) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a latent class model with covariates, where the predicted latent classes from 2 act as the only categorical variable available, and its probability mass function within each class is fixed and equal to the classification error. This implementation requires the function `correction_em()` in `LCA-Covariates-Algorithms.R`.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_corrected <- rep(0, Rep_Tot)

# Define useful quantities to compute the full--model log-likelihood sequence.
f_cheating_3_step <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
nclass = 2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

# Perform the three step algorithm.
time_3_step_corrected <- system.time(
  
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
fit_unconditional <- unconditional_em(f_cheating_unconditional, 
                                      cheating, nclass = 2, seed = seed_rep[rep])

#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment and compute classification error
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]], 1, which.max)
class_err <- matrix(0, nclass, nclass)
rownames(class_err) <- paste("W", c(1:nclass), sep="")
colnames(class_err) <- paste("X", c(1:nclass), sep="")
for (r in 1:nclass){
class_err[,r] <- (t(dummy(pred_class))%*%as.matrix(fit_unconditional[[4]][,r],
                  dim(cheating)[1],1)/dim(cheating)[1])/fit_unconditional[[5]][1,r]}
class_err <- t(class_err)

#---------------------------------------------------------------------------------------------------
# 3] estimate the beta coefficient from the correction procedure in Vermunt (2010)
#---------------------------------------------------------------------------------------------------
f_cheating_3_step_correct <- cbind(pred_class) ~ GPA
fit_correct <- correction_em(f_cheating_3_step_correct, cheating, seed = seed_rep[rep],
                             classification_error = class_err)

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(fit_correct[[3]], x_cheating_3_step, R_cheating_3_step)
llik_3_step_corrected[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],
                                  y_cheating_3_step))))}

)[3]
```

#### 5. Nested EM algorithm (one-step maximization)

We now implement our **nested EM** algorithm for improved one-step estimation of latent class models with covariates. This routine is carefully described in Section 2.2 of our paper, and leverages the recently developed Pòlya-Gamma data augmentation ([Polson et al. 2013](http://www.tandfonline.com/doi/abs/10.1080/01621459.2013.829001)). The implementation requires the function `nested_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create allocation matrices for the quantities to be monitored, as above.
iter_NEM <- rep(0, Rep_Tot)
llik_NEM <- matrix(0, Rep_Tot, 1000)
llik_decrement_NEM <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NEM <- system.time(    
  
for (rep in 1:Rep_Tot){
fit_NEM <- nested_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep])
iter_NEM[rep] <- fit_NEM[[1]]   
llik_NEM[rep,] <- fit_NEM[[2]]
llik_decrement_NEM[rep,] <- fit_NEM[[3]]}

)[3]
```

#### 6. Hybrid nested EM algorithm (one-step maximization)

Here we consider a more efficient hybrid version of the **nested EM** algorithm which reaches a neighborhood of the maximum using the more stable **nested EM**, and then switches to Newton-Raphson methods to speed convergence. This routine is carefully described in Section 3.3 of our paper, and requires the function `hybrid_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create allocation matrices for the quantities to be monitored, as above.
iter_HYB <- rep(0, Rep_Tot)
llik_HYB <- matrix(0, Rep_Tot, 1000)
llik_decrement_HYB <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_HYB <- system.time(
  
for (rep in 1:Rep_Tot){
fit_HYB <- hybrid_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], epsilon = 0.1)
iter_HYB[rep] <- fit_HYB[[1]]   
llik_HYB[rep,] <- fit_HYB[[2]]
llik_decrement_HYB[rep,] <- fit_HYB[[3]]}

)[3]
```

### Performance comparison

Once the parameters have been estimated under the computational routines implemented above, we compare the maximization performance and the computational efficiency of the different algorithms, in order to reproduce the results in Table 1 of our paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). In particular, we consider the following quantities, computed for each run of every routine:

**Maximization Performance**

-   Number of runs with a drop in the log-likelihood sequence.
-   Number of runs converging to values which are not the maximum log-likelihood.
-   For the runs reaching a local mode, we also compute the quartiles of the difference between the log-likelihoods in the local modes and the maximum one.

**Computational Efficiency**

-   Number of iterations for convergence, computed only for the runs reaching the maximum log-likelihood.
-   Averaged computational time for each run.

Consistent with the above goal, let us create a function which computes the measures of performance from the output of the different algorithms.

``` r
performance_algo <- function(max_loglik, n_rep, loglik_seq, loglik_decay, n_iter, time, delta){

#----------------------------------------------------------------------------------------------
# Number of runs with a drop in the log-likelihood sequence.
#----------------------------------------------------------------------------------------------  
n_drops <- 0
for (rep in 1:n_rep){
n_drops <- n_drops + (sum(loglik_decay[rep,1:n_iter[rep]]) > 0)*1}

#----------------------------------------------------------------------------------------------
# Number of runs converging to values which are not the maximum log-likelihood.
#---------------------------------------------------------------------------------------------- 
n_l_modes <- sum(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) > delta)

#----------------------------------------------------------------------------------------------
# Quantiles difference between log-likelihood in local modes and maximum one.
#----------------------------------------------------------------------------------------------
any_mode <- sum((abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) > delta)*1)
if (any_mode > 0){ 
sel_modes <- which(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) > delta)
diff_llik <- quantile(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)])[sel_modes])[2:4]} else {
diff_llik <- rep(0,3)  
}

#----------------------------------------------------------------------------------------------
# Quantiles iterations for convergence to maximum log-likelihood.
#----------------------------------------------------------------------------------------------
sel_convergence <- which(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) <= delta)
iter_convergence <- quantile(n_iter[sel_convergence])[2:4]

#----------------------------------------------------------------------------------------------
# Averaged computational time for each iteration.
#----------------------------------------------------------------------------------------------
averaged_time <- time/n_rep

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#Create the vector with the measures to be saved
output<-c(n_drops, n_l_modes, diff_llik, iter_convergence, averaged_time)
return(output)
}
```

In reproducing the results in Table 1, let us first define the correct maximum log-likelihood `max_llik`, and a control quantity `delta` defining the minimum deviation from `max_llik` which is indicative of a local mode.

``` r
max_llik <- c(-429.6384)
delta <- 0.01
```

Finally, let us create a matrix `Table_Performance` which contains the performance measures for the different algorithms.

``` r
Table_Performance <- matrix(NA,9,8)

rownames(Table_Performance) <- c("N. Decays",
                                 "N. Local Modes",
                                 "Q1 Log-L in Local Modes",
                                 "Q2 Log-L in Local Modes",
                                 "Q3 Log-L in Local Modes",
                                 "Q1 N. Iterat. Converge max(Log-L)",
                                 "Q2 N. Iterat. Converge max(Log-L)",
                                 "Q3 N. Iterat. Converge max(Log-L)",
                                 "Averaged Time")

colnames(Table_Performance) <- c("NR EM 1","NR EM 0.75","NR EM 0.5","NR EM 0.25","CLASSIC. 3-STEP",
                                 "CORREC. 3-STEP","NESTED EM","HYBRID EM")
```

We can now compute the different performance measures for our algorithms.

**1. Performance EM algorithm with Newton-Raphson methods *α* = 1 (one-step maximization)**

``` r
Table_Performance[,1] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_1, 
                                          llik_decrement_NR_EM_alpha_1, iter_NR_EM_alpha_1, 
                                          time_NR_EM_alpha_1, delta)
```

**2.1 Performance EM algorithm with Newton-Raphson methods *α* = 0.75 (one-step maximization)**

``` r
Table_Performance[,2] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.75, 
                                          llik_decrement_NR_EM_alpha_0.75, iter_NR_EM_alpha_0.75, 
                                          time_NR_EM_alpha_0.75, delta)
```

**2.2 Performance EM algorithm with Newton-Raphson methods *α* = 0.5 (one-step maximization)**

``` r
Table_Performance[,3] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.5, 
                                          llik_decrement_NR_EM_alpha_0.5, iter_NR_EM_alpha_0.5, 
                                          time_NR_EM_alpha_0.5, delta)
```

**2.3 Performance EM algorithm with Newton-Raphson methods *α* = 0.25 (one-step maximization)**

``` r
Table_Performance[,4] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.25, 
                                          llik_decrement_NR_EM_alpha_0.25, iter_NR_EM_alpha_0.25, 
                                          time_NR_EM_alpha_0.25, delta)
```

**3 Performance Classical 3-step algorithm (three-step maximization)** &gt; As discussed in the paper, since all the three-step runs converge systematically to local modes, we do not study the number of iterations to reach convergence. In fact, these routines never converge to the maximum log-likelihood. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three-step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full-model log-likelihood.

``` r
Table_Performance[,5] <- performance_algo(max_llik, Rep_Tot, llik_3_step_classical, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_classical, delta)
```

**4 Performance Bias-corrected 3-step algorithm (three-step maximization)** &gt; Even in this case all the runs converge systematically to local modes. Therefore we do not study the number of iterations to reach convergence. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three-step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full-model log-likelihood.

``` r
Table_Performance[,6] <- performance_algo(max_llik, Rep_Tot, llik_3_step_corrected, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_corrected, delta)
```

**5 Performance nested EM algorithm (one-step maximization)**

``` r
Table_Performance[,7] <- performance_algo(max_llik, Rep_Tot, llik_NEM, llik_decrement_NEM, 
                                          iter_NEM, time_NEM, delta)
```

**6 Performance hybrid nested EM algorithm (one-step maximization)**

``` r
Table_Performance[,8] <- performance_algo(max_llik, Rep_Tot, llik_HYB, llik_decrement_HYB, 
                                          iter_HYB, time_HYB, delta)
```

Analysis of the output from the table
-------------------------------------

Let us finally visualize the performance table, which reproduces Table 1 in the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864).

In particular, the maximization performance and the computational efficiency of the EM algorithm with one Newton-Raphson step, along with those of the re-scaled modifications, are:

``` r
library(knitr)
kable(Table_Performance[,1:4])
```

|                                   |    NR EM 1|  NR EM 0.75|  NR EM 0.5|  NR EM 0.25|
|-----------------------------------|----------:|-----------:|----------:|-----------:|
| N. Decays                         |   65.00000|    42.00000|   23.00000|    10.00000|
| N. Local Modes                    |   66.00000|    46.00000|   26.00000|     8.00000|
| Q1 Log-L in Local Modes           |   24.02133|    18.48638|   24.02133|    16.80014|
| Q2 Log-L in Local Modes           |   24.24113|    24.02133|   24.02133|    24.02133|
| Q3 Log-L in Local Modes           |   35.59522|    35.59522|   35.59522|    26.79844|
| Q1 N. Iterat. Converge max(Log-L) |  105.50000|   114.00000|  145.00000|   233.75000|
| Q2 N. Iterat. Converge max(Log-L) |  114.00000|   125.50000|  152.00000|   240.50000|
| Q3 N. Iterat. Converge max(Log-L) |  127.00000|   137.00000|  162.75000|   252.00000|
| Averaged Time                     |    0.03696|     0.04062|    0.05586|     0.10355|

The maximization performance and the computational efficiency of the three-step estimation algorithms, along with those of our **nested EM** and its hybrid modification, are instead:

``` r
library(knitr)
kable(Table_Performance[,5:8])
```

|                                   |  CLASSIC. 3-STEP|  CORREC. 3-STEP|  NESTED EM|  HYBRID EM|
|-----------------------------------|----------------:|---------------:|----------:|----------:|
| N. Decays                         |         0.000000|       0.0000000|    0.00000|    0.00000|
| N. Local Modes                    |       100.000000|     100.0000000|    0.00000|    0.00000|
| Q1 Log-L in Local Modes           |         1.632104|       0.5393288|    0.00000|    0.00000|
| Q2 Log-L in Local Modes           |         1.632106|       0.5393291|    0.00000|    0.00000|
| Q3 Log-L in Local Modes           |         1.632107|       0.5393402|    0.00000|    0.00000|
| Q1 N. Iterat. Converge max(Log-L) |               NA|              NA|  178.00000|  130.75000|
| Q2 N. Iterat. Converge max(Log-L) |               NA|              NA|  184.50000|  135.50000|
| Q3 N. Iterat. Converge max(Log-L) |               NA|              NA|  189.00000|  140.00000|
| Averaged Time                     |         0.205970|       0.1842300|    0.09167|    0.05819|

Reproduce the left plot in Figure 2 of the paper
------------------------------------------------

We conclude the analysis by providing the code to obtain the left plot in Figure 2 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). This plot compares, for a selected run `sel <- 24`, the log-likelihood sequence obtained under our **nested EM** with the one provided by the standard EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407).

Let us first load some useful libraries and choose the run on which to focus.

``` r
library(ggplot2)
library(reshape)

sel <- 24
```

We then create a dataset `data_plot` with two columns. The first contains the log-likelihood sequence of the **nested EM**, whereas the second comprises the one from the standard EM algorithm with Newton-Raphson methods.

``` r
data_plot <- cbind(llik_NEM[sel,],llik_NR_EM_alpha_1[sel,])
data_plot <- data_plot[c(1:max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),]
data_plot[c((iter_NR_EM_alpha_1[sel] + 1):
          max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),2] <- data_plot[iter_NR_EM_alpha_1[sel],2]
```

Finally we create the Figure.

``` r
data_ggplot <- melt(data_plot)
data_ggplot <- as.data.frame(data_ggplot)
data_ggplot$app <- "CHEATING DATA"

plot <- ggplot(data = data_ggplot, aes(x = X1, y = value, group = X2)) +
               geom_line(aes(linetype = as.factor(X2))) +
               coord_cartesian(ylim = c(-800, -420), xlim = c(0, 50)) + theme_bw() +
               labs(x = "iteration", y = expression(paste(l,"(", theta^{(t)}, ";",x,",",y,")"))) +
               theme(legend.position = "none", strip.text.x = element_text(size = 11, face = "bold")) +
               facet_wrap(~ app)

plot
```

![](Cheating-Data_files/figure-markdown_github/unnamed-chunk-29-1.png)
