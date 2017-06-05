Performance assessment for the CHEATING data
================
Daniele Durante

### Description
================

This tutorial implementation focuses on assessing the maximization performance, and the computational efficiency of the different algorithms for the estimation of latent class models with covariates. In particular this assessment considers the dataset `cheating` from the `R` library [`poLCA`](https://www.jstatsoft.org/article/view/v042i10).

The analyses reproduce those discussed in Section 3.1 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864) where we propose a novel **nested EM** algorithm for improved maximum likelihood estimation of latent class models with covariates.

### Load the source functions and the data

The functions for the implementation of the different algorithms---including the **nested EM**, and the popular competitors currently considered in the literature---are available in the source file [`LCA-Covariates-Algorithms.R`](https://github.com/danieledurante/nEM/blob/master/LCA-Covariates-Algorithms.R). More comments on the different maximization routines can be found in the file `LCA-Covariates-Algorithms.R`.

Let us load this source file, along with the `cheating` dataset.

``` r
rm(list=ls())
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

Since the main focus in on comparing the computational performance of the different algorithms---instead of providing inference on the `cheating` data---let us for simplicity remove all the statistical units having missing values.

``` r
cheating <- na.omit(cheating)
```

We aim to estimate the latent class model for the `J = 4` different cheating behaviors, having the variable `GPA` (grade point average) as a covariate in the multinomial logistic regression for the latent classes. Using the syntax of the library `poLCA`, this model can be defined as follows:

``` r
f_cheating <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM) ~ GPA
```

As discussed in Section 3 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), to provide a detailed computational assessment, we perform estimation under the different algorithms for `Rep_Tot = 100` runs at varying initializations. For each run, the algorithms are all initialized at the same starting values, which are controlled by a `seed` (changing across the different runs). Let us therefore define this 100 seeds.

``` r
Rep_Tot <- 100
seed_rep <- c(101:200)
seed_rep[35] <- 1
seed_rep[89] <- 2
```

Note that in the above `seed_rep` specification, some values are tuned since the one--step EM algorithm incorporating Newton-Raphson methods converged to undefined log-likelihoods in some runs. Hence we changed some seeds to improve the behavior of this competing algorithm.

### Estimation under the different maximization routines

We perform estimation of the parameters in the latent class regression model with covariates defined above, under different computational routines---including our novel **nested EM** algorithm---and compare maximization performance, along with computational efficiency.

Consistent with the tutorial analyses in [Linzer and Lewis (2011)](https://www.jstatsoft.org/article/view/v042i10), we focus on a model with `R=2` latent classes.

#### 1. EM algorithm with Newton-Raphson methods (one--step maximization)

Here we consider the one--step EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407), and discussed in Section 1.1 of our paper. This requires the function `newton_em()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before.

Let us first create the quantities to be monitored for each run. These include the number of iterations to reach convergence, the log-likelihood sequence, and a vector monitoring presence (1) or absence (0) of drops in the log-likelihood sequence.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_1 <- rep(0,Rep_Tot)
llik_NR_EM_alpha_1 <- matrix(0,Rep_Tot,1000)
llik_decrement_NR_EM_alpha_1 <- matrix(0,Rep_Tot,1000)

# 1000 means that the maximum number of iteration of the EM we will consider is 1000. 
```

Finally let us perform the `Rep_Tot = 100` runs of the one--step EM algorithm with Newton-Raphson methods. We also monitor the computational time via the function `system.time()`.

``` r
# Perform the algorithm.
time_NR_EM_alpha_1 <- system.time(
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep])
iter_NR_EM_alpha_1[rep] <- fit_NR_EM[[1]]   
llik_NR_EM_alpha_1[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_1[rep,] <- fit_NR_EM[[3]]})[3]
```

#### 2. Re--scaled EM algorithm with Newton-Raphson methods (one--step maximization)

Here we consider the re--scaled version of the above one--step EM algorithm with Newton-Raphson methods. This modification is discussed in Section 1.1 of our paper and its general version can be found in Chapter 1.5.6 of [McLachlan and Krishnan (2007)](http://onlinelibrary.wiley.com/book/10.1002/9780470191613). Also this algorithm requires the function `newton_em()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before. However now the parameter 0 &lt; *α* &lt; 1 should be modified to reduce concerns about drops in the log-likelihood sequence. Here we consider:

-   The case *α* = 0.75.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.75 <- rep(0,Rep_Tot)
llik_NR_EM_alpha_0.75 <- matrix(0,Rep_Tot,1000)
llik_decrement_NR_EM_alpha_0.75 <- matrix(0,Rep_Tot,1000)

# Perform the algorithm.
time_NR_EM_alpha_0.75 <- system.time(   
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep],alpha=0.75)
iter_NR_EM_alpha_0.75[rep] <- fit_NR_EM[[1]]    
llik_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[3]]})[3]
```

-   The case *α* = 0.50.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.5 <- rep(0,Rep_Tot)
llik_NR_EM_alpha_0.5 <- matrix(0,Rep_Tot,1000)
llik_decrement_NR_EM_alpha_0.5 <- matrix(0,Rep_Tot,1000)

# Perform the algorithm.
time_NR_EM_alpha_0.5 <- system.time(    
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep],alpha=0.5)
iter_NR_EM_alpha_0.5[rep] <- fit_NR_EM[[1]] 
llik_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[3]]})[3]
```

-   The case *α* = 0.25.

``` r
# Create allocation matrices for the quantities to be monitored.
iter_NR_EM_alpha_0.25 <- rep(0,Rep_Tot)
llik_NR_EM_alpha_0.25 <- matrix(0,Rep_Tot,1000)
llik_decrement_NR_EM_alpha_0.25 <- matrix(0,Rep_Tot,1000)

# Perform the algorithm.
time_NR_EM_alpha_0.25 <- system.time(   
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep],alpha=0.25)
iter_NR_EM_alpha_0.25[rep] <- fit_NR_EM[[1]]    
llik_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[3]]})[3]
```

#### 3. Classical 3--step algorithm (three--step maximization)

Here we consider the classical three--step strategy to estimate latent class models with covariates (e.g. [Clogg 1995](https://www.iser.essex.ac.uk/research/publications/494549)). As discussed in Section 1.2 of our paper, this algorithm consists of the following three steps.

1.  Estimate a latent class model without covariates. This requires the function `unconditional_em()` in `LCA-Covariates-Algorithms.R`---applied to the model `f_cheating_unconditional <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~1`.
2.  Using the estimates in 1, predict the latent classes $\\hat{s}\_i$, *i* = 1, ..., *n*, by assigning each unit *i* to the class *r* with the highest pr$(s\_i=r | \\hat{\\pi}, \\hat{\\nu},y\_i)$.
3.  Estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a multinomial logistic regression with $\\hat{s}\_1,...,\\hat{s}\_n$ as responses---using the `R` function `multinom` in the library `nnet`.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_classical <- rep(0,Rep_Tot)

# Define useful quantities to compute the full--model log-likelihood sequence at the end of the routine.
f_cheating_3_step <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~GPA
nclass=2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

# Perform the three step algorithm.
time_3_step_classical<-system.time(
for (rep in 1:Rep_Tot){
#----------------------------------------------------------------------------------------------
# 1] ESTIMATE AN UNCONDITIONAL LATENT CLASS MODEL   
#----------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~1
fit_unconditional <- unconditional_em(f_cheating_unconditional,cheating,nclass=2,seed=seed_rep[rep])
#----------------------------------------------------------------------------------------------
# 2] PREDICT THE CLASS OF EACH UNIT VIA MODAL ASSIGNMENT    
#----------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]],1,which.max)
#----------------------------------------------------------------------------------------------
# 3] ESTIMATE THE BETA COEFFICIENTS VIA MULTINOMIAL LOGIT WITH PREDICTED CLASSES AS RESPONSES
#----------------------------------------------------------------------------------------------
b <- c(t(summary(multinom(pred_class~cheating$GPA,trace=FALSE))$coefficients))

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(b, x_cheating_3_step, R_cheating_3_step)
llik_3_step_classical[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],y_cheating_3_step))))})[3]
```

#### 4. Bias--corrected 3--step algorithm (three--step maximization)

Here we implement the modification proposed by [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) of the classical three--step methods, in order to reduce the bias of the estimators. This strategy is discussed in Sections 1.2 and 4 of our paper, and proceed as follows:

1.  Estimate a latent class model without covariates. This requires the function `unconditional_em()` in `LCA-Covariates-Algorithms.R`---applied to the model `f_cheating_unconditional <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~1`.
2.  Using the estimates in 1, predict the latent classes $\\hat{s}\_i$, *i* = 1, ..., *n*, by assigning unit each *i* to the class *r* with the highest pr$(s\_i=r | \\hat{\\pi}, \\hat{\\nu},y\_i)$. Compute also the classification error by applying equation (6) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved).
3.  Following equation (19) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a latent class model with covariates, where the predicted latent classes from 2 act as the only categorical variable available, and its probability mass function within each class is fixed and equal to the classification error. This implementation requires the function `correction_em()` in `LCA-Covariates-Algorithms.R`.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_corrected <- rep(0,Rep_Tot)

# Define useful quantities to compute the full--model log-likelihood sequence at the end of the routine.
f_cheating_3_step <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~GPA
nclass=2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

# Perform the three step algorithm.
time_3_step_corrected<-system.time(
for (rep in 1:Rep_Tot){
#----------------------------------------------------------------------------------------------
# 1] ESTIMATE AN UNCONDITIONAL LATENT CLASS MODEL   
#----------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM)~1
fit_unconditional<-unconditional_em(f_cheating_unconditional,cheating,nclass=2,seed=seed_rep[rep])
#----------------------------------------------------------------------------------------------
# 2] PREDICT THE CLASS OF EACH UNIT VIA MODAL ASSIGNMENT AND COMPUTE CLASSIFICATION ERROR
#----------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]],1,which.max)
class_err <- matrix(0,nclass,nclass)
rownames(class_err) <- paste("W",c(1:nclass),sep="")
colnames(class_err) <- paste("X",c(1:nclass),sep="")
for (r in 1:nclass){
class_err[,r] <- (t(dummy(pred_class))%*%as.matrix(fit_unconditional[[4]][,r],dim(cheating)[1],1)/dim(cheating)[1])/fit_unconditional[[5]][1,r]}
class_err <- t(class_err)
#----------------------------------------------------------------------------------------------
# 3] ESTIMATE THE BETA COEFFICIENTS VIA THE CORRECTION PROCEDURE PROPOSED IN VERMUNT (2010) 
#----------------------------------------------------------------------------------------------
f_cheating_3_step_correct <- cbind(pred_class)~GPA
fit_correct <- correction_em(f_cheating_3_step_correct,cheating,seed=seed_rep[rep],classification_error = class_err)

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(fit_correct[[3]], x_cheating_3_step, R_cheating_3_step)
llik_3_step_corrected[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],y_cheating_3_step))))})[3]
```

#### 5. Nested EM algorithm (one--step maximization)

We now implement our **nested EM** algorithm for improved one--step estimation of latent class models with covariates. This routine is carefully described in Section 2.2 of our paper, and leverages the recently developed Pòlya-Gamma data augmentation ([Polson et al. 2013](http://www.tandfonline.com/doi/abs/10.1080/01621459.2013.829001)). The implementation requires the function `nested_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create allocation matrices for the quantities to be monitored, as above.
iter_NEM <- rep(0,Rep_Tot)
llik_NEM <- matrix(0,Rep_Tot,1000)
llik_decrement_NEM <- matrix(0,Rep_Tot,1000)

# Perform the algorithm.
time_NEM <- system.time(    
for (rep in 1:Rep_Tot){
fit_NEM <- nested_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep])
iter_NEM[rep] <- fit_NEM[[1]]   
llik_NEM[rep,] <- fit_NEM[[2]]
llik_decrement_NEM[rep,] <- fit_NEM[[3]]})[3]
```

#### 6. Hybrid nested EM algorithm (one--step maximization)

Here we consider a more efficient hybrid version of the **nested EM** algorithm which reaches a neighborhood of the maximum using the more stable **nested EM**, and then switches to Newton-Raphson methods to speed convergence. This routine is carefully described in Section 3.3 of our paper, and requires the function `hybrid_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create allocation matrices for the quantities to be monitored, as above.
iter_HYB <- rep(0,Rep_Tot)
llik_HYB <- matrix(0,Rep_Tot,1000)
llik_decrement_HYB <- matrix(0,Rep_Tot,1000)

# Perform the algorithm.
time_HYB <- system.time(    
for (rep in 1:Rep_Tot){
fit_HYB <- hybrid_em(f_cheating,cheating,nclass=2,seed=seed_rep[rep],epsilon=0.1)
iter_HYB[rep] <- fit_HYB[[1]]   
llik_HYB[rep,] <- fit_HYB[[2]]
llik_decrement_HYB[rep,] <- fit_HYB[[3]]})[3]
```

### Performance comparison

Once the parameters have been estimated under the computational routines implemented above, we compare the maximization performance, and the computational efficiency of the different algorithms, in order to reproduce the results in Table 1 of our paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). In particular, we consider the following quantities---computed for each run of every routine.

**Maximization Performance**

-   Number of runs with a drop in the log-likelihood sequence.
-   Number of runs converging to values which are not the maximum log-likelihood.
-   For the runs reaching a local mode, we also compute the quartiles of the difference between the log-likelihood in the local modes and the maximum one.

**Computational Efficiency**

-   Number of iterations for convergence, computed only for the runs reaching the maximum log-likelihood.
-   Averaged computational time for each run.

In reproducing the results in Table 1, let us first define the correct maximum log-likelihood `max_llik`:

``` r
max_llik <- c(-429.6384)
```

and let us also create a matrix with the different measures on the rows, and the algorithms under analysis on the columns.

``` r
Table_Performance <- matrix(NA,9,8)

rownames(Table_Performance) <- c("Number Decays","Number Local Modes","Q1 Log-L in Local Modes","Q2 Log-L in Local Modes","Q3 Log-L in Local Modes","Q1 Number Iteration Convergence max(Log-L)","Q2 Number Iteration Convergence max(Log-L)","Q3 Number Iteration Convergence max(Log-L)","Averaged Time")
colnames(Table_Performance) <- c("NR EM 1","NR EM 0.75","NR EM 0.5","NR EM 0.25","CLASSICAL 3-STEP","CORRECTED 3-STEP","NESTED EM","HYBRID EM")
```

To find the local modes we need to choose a control quantity `delta` defining the maximum deviation from `max_llik` which is indicative of a local mode.

``` r
delta <- 0.01
```

We can now compute the different performance measures for our algorithms.

**1. Performance EM algorithm with Newton-Raphson methods *α* = 1 (one--step maximization)**

``` r
decr_llik_a_1 <- 0
for (rep in 1:Rep_Tot){
decr_llik_a_1 <- decr_llik_a_1+(sum(llik_decrement_NR_EM_alpha_1[rep,1:iter_NR_EM_alpha_1[rep]])>0)*1}
Table_Performance[1,1] <- decr_llik_a_1
```

``` r
Table_Performance[2,1] <- sum(abs(max_llik-llik_NR_EM_alpha_1[cbind(1:100,iter_NR_EM_alpha_1)])>delta)
```

``` r
Table_Performance[3:5,1] <- quantile(abs(max_llik-llik_NR_EM_alpha_1[cbind(1:100,iter_NR_EM_alpha_1)])[which(abs(max_llik-llik_NR_EM_alpha_1[cbind(1:100,iter_NR_EM_alpha_1)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,1] <- quantile(iter_NR_EM_alpha_1[which(abs(max_llik-llik_NR_EM_alpha_1[cbind(1:100,iter_NR_EM_alpha_1)])<=delta)])[2:4]
```

``` r
Table_Performance[9,1]<-time_NR_EM_alpha_1/Rep_Tot
```

**2.1 Performance EM algorithm with Newton-Raphson methods *α* = 0.75 (one--step maximization)**

``` r
decr_llik_a_0.75 <- 0
for (rep in 1:Rep_Tot){
decr_llik_a_0.75 <- decr_llik_a_0.75+(sum(llik_decrement_NR_EM_alpha_0.75[rep,1:iter_NR_EM_alpha_0.75[rep]])>0)*1}
Table_Performance[1,2]<-decr_llik_a_0.75
```

``` r
Table_Performance[2,2] <- sum(abs(max_llik-llik_NR_EM_alpha_0.75[cbind(1:100,iter_NR_EM_alpha_0.75)])>delta)
```

``` r
Table_Performance[3:5,2] <- quantile(abs(max_llik-llik_NR_EM_alpha_0.75[cbind(1:100,iter_NR_EM_alpha_0.75)])[which(abs(max_llik-llik_NR_EM_alpha_0.75[cbind(1:100,iter_NR_EM_alpha_0.75)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,2] <- quantile(iter_NR_EM_alpha_0.75[which(abs(max_llik-llik_NR_EM_alpha_0.75[cbind(1:100,iter_NR_EM_alpha_0.75)])<=delta)])[2:4]
```

``` r
Table_Performance[9,2] <- time_NR_EM_alpha_0.75/Rep_Tot
```

**2.2 Performance EM algorithm with Newton-Raphson methods *α* = 0.5 (one--step maximization)**

``` r
decr_llik_a_0.5<-0
for (rep in 1:Rep_Tot){
decr_llik_a_0.5 <- decr_llik_a_0.5+(sum(llik_decrement_NR_EM_alpha_0.5[rep,1:iter_NR_EM_alpha_0.5[rep]])>0)*1}
Table_Performance[1,3] <- decr_llik_a_0.5
```

``` r
Table_Performance[2,3] <- sum(abs(max_llik-llik_NR_EM_alpha_0.5[cbind(1:100,iter_NR_EM_alpha_0.5)])>delta)
```

``` r
Table_Performance[3:5,3] <- quantile(abs(max_llik-llik_NR_EM_alpha_0.5[cbind(1:100,iter_NR_EM_alpha_0.5)])[which(abs(max_llik-llik_NR_EM_alpha_0.5[cbind(1:100,iter_NR_EM_alpha_0.5)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,3] <- quantile(iter_NR_EM_alpha_0.5[which(abs(max_llik-llik_NR_EM_alpha_0.5[cbind(1:100,iter_NR_EM_alpha_0.5)])<=delta)])[2:4]
```

``` r
Table_Performance[9,3] <- time_NR_EM_alpha_0.5/Rep_Tot
```

**2.3 Performance EM algorithm with Newton-Raphson methods *α* = 0.25 (one--step maximization)**

``` r
decr_llik_a_0.25<-0
for (rep in 1:Rep_Tot){
decr_llik_a_0.25 <- decr_llik_a_0.25+(sum(llik_decrement_NR_EM_alpha_0.25[rep,1:iter_NR_EM_alpha_0.25[rep]])>0)*1}
Table_Performance[1,4] <- decr_llik_a_0.25
```

``` r
Table_Performance[2,4] <- sum(abs(max_llik-llik_NR_EM_alpha_0.25[cbind(1:100,iter_NR_EM_alpha_0.25)])>delta)
```

``` r
Table_Performance[3:5,4] <- quantile(abs(max_llik-llik_NR_EM_alpha_0.25[cbind(1:100,iter_NR_EM_alpha_0.25)])[which(abs(max_llik-llik_NR_EM_alpha_0.25[cbind(1:100,iter_NR_EM_alpha_0.25)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,4] <- quantile(iter_NR_EM_alpha_0.25[which(abs(max_llik-llik_NR_EM_alpha_0.25[cbind(1:100,iter_NR_EM_alpha_0.25)])<=delta)])[2:4]
```

``` r
Table_Performance[9,4] <- time_NR_EM_alpha_0.25/Rep_Tot
```

**3 Performance Classical 3--step algorithm (three--step maximization)** As discussed in the paper, since all the three--step runs converge systematically to local modes, we do not study the number of iterations to reach convergence. In fact, these routines never converge to the maximum log-likelihood. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three--step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full--model log-likelihood.

``` r
Table_Performance[2,5] <- sum(abs(max_llik-llik_3_step_classical)>delta)
```

``` r
Table_Performance[3:5,5] <- quantile(abs(max_llik-llik_3_step_classical)[which(abs(max_llik-llik_3_step_classical)>delta)])[2:4]
```

``` r
Table_Performance[9,5] <- time_3_step_classical/Rep_Tot
```

**4 Performance Bias--corrected 3--step algorithm (three--step maximization)** Even in this case all the runs converge systematically to local modes. Therefore we do not study the number of iterations to reach convergence. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three--step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full--model log-likelihood.

``` r
Table_Performance[2,6] <- sum(abs(max_llik-llik_3_step_corrected)>delta)
```

``` r
Table_Performance[3:5,6] <- quantile(abs(max_llik-llik_3_step_corrected)[which(abs(max_llik-llik_3_step_corrected)>delta)])[2:4]
```

``` r
Table_Performance[9,6] <- time_3_step_corrected/Rep_Tot
```

**5 Performance nested EM algorithm (one--step maximization)**

``` r
decr_llik_NEM <- 0
for (rep in 1:Rep_Tot){
decr_llik_NEM <- decr_llik_NEM+(sum(llik_decrement_NEM[rep,1:iter_NEM[rep]])>0)*1}
Table_Performance[1,7] <- decr_llik_NEM
```

``` r
Table_Performance[2,7] <- sum(abs(max_llik-llik_NEM[cbind(1:100,iter_NEM)])>delta)
```

``` r
Table_Performance[3:5,7] <- quantile(abs(max_llik-llik_NEM[cbind(1:100,iter_NEM)])[which(abs(max_llik-llik_NEM[cbind(1:100,iter_NEM)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,7] <- quantile(iter_NEM[which(abs(max_llik-llik_NEM[cbind(1:100,iter_NEM)])<=delta)])[2:4]
```

``` r
Table_Performance[9,7] <- time_NEM/Rep_Tot
```

**6 Performance hybrid nested EM algorithm (one--step maximization)**

``` r
decr_llik_HYB <- 0
for (rep in 1:Rep_Tot){
decr_llik_HYB <- decr_llik_HYB+(sum(llik_decrement_HYB[rep,1:iter_HYB[rep]])>0)*1}
Table_Performance[1,8] <- decr_llik_HYB
```

``` r
Table_Performance[2,8] <- sum(abs(max_llik-llik_HYB[cbind(1:100,iter_HYB)])>delta)
```

``` r
Table_Performance[3:5,8] <- quantile(abs(max_llik-llik_HYB[cbind(1:100,iter_HYB)])[which(abs(max_llik-llik_HYB[cbind(1:100,iter_HYB)])>delta)])[2:4]
```

``` r
Table_Performance[6:8,8] <- quantile(iter_HYB[which(abs(max_llik-llik_HYB[cbind(1:100,iter_HYB)])<=delta)])[2:4]
```

``` r
Table_Performance[9,8] <- time_HYB/Rep_Tot
```

### Analysis of the output from the table

Let us finally visualize the performance table, which reproduces Table 1 in the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864).

In particular, the maximization performance and the computational efficiency of the EM algorithm with one Newton-Raphson step, along with those of the re--scaled modifications are:

``` r
library(knitr)
kable(Table_Performance[,1:4])
```

|                                            |    NR EM 1|  NR EM 0.75|  NR EM 0.5|  NR EM 0.25|
|--------------------------------------------|----------:|-----------:|----------:|-----------:|
| Number Decays                              |   65.00000|    42.00000|   23.00000|    10.00000|
| Number Local Modes                         |   66.00000|    46.00000|   26.00000|     8.00000|
| Q1 Log-L in Local Modes                    |   24.02133|    18.48638|   24.02133|    16.80014|
| Q2 Log-L in Local Modes                    |   24.24113|    24.02133|   24.02133|    24.02133|
| Q3 Log-L in Local Modes                    |   35.59522|    35.59522|   35.59522|    26.79844|
| Q1 Number Iteration Convergence max(Log-L) |  105.50000|   114.00000|  145.00000|   233.75000|
| Q2 Number Iteration Convergence max(Log-L) |  114.00000|   125.50000|  152.00000|   240.50000|
| Q3 Number Iteration Convergence max(Log-L) |  127.00000|   137.00000|  162.75000|   252.00000|
| Averaged Time                              |    0.02995|     0.04521|    0.05578|     0.09749|

The maximization performance and the computational efficiency of the three--step estimation algorithms, along with those of our **nested EM** and its hybrid modification are instead:

``` r
library(knitr)
kable(Table_Performance[,5:8])
```

|                                            |  CLASSICAL 3-STEP|  CORRECTED 3-STEP|  NESTED EM|  HYBRID EM|
|--------------------------------------------|-----------------:|-----------------:|----------:|----------:|
| Number Decays                              |                NA|                NA|    0.00000|    0.00000|
| Number Local Modes                         |        100.000000|       100.0000000|    0.00000|    0.00000|
| Q1 Log-L in Local Modes                    |          1.632104|         0.5393288|         NA|         NA|
| Q2 Log-L in Local Modes                    |          1.632106|         0.5393291|         NA|         NA|
| Q3 Log-L in Local Modes                    |          1.632107|         0.5393402|         NA|         NA|
| Q1 Number Iteration Convergence max(Log-L) |                NA|                NA|  178.00000|  130.75000|
| Q2 Number Iteration Convergence max(Log-L) |                NA|                NA|  184.50000|  135.50000|
| Q3 Number Iteration Convergence max(Log-L) |                NA|                NA|  189.00000|  140.00000|
| Averaged Time                              |          0.201100|         0.2008300|    0.10004|    0.06854|

### Reproduce the left plot in Figure 2 of the paper

We conclude the analysis by providing the code to obtain the left plot in Figure 2 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). This plot compares, for a selected run `sel <- 24`, the log-likelihood sequence obtained under our **nested EM** with the one provided by thestandard EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407).

Let us first load some useful libraries and choose the run on which to focus.

``` r
library(ggplot2)
library(reshape)

sel<-24
```

We then create a dataset with two columns. The first contains the log-likelihood sequence of the **nested EM**, whereas the second comprises the one from the standard EM algorithm with Newton-Raphson methods.

``` r
data_plot<-cbind(llik_NEM[sel,],llik_NR_EM_alpha_1[sel,])
data_plot<-data_plot[c(1:max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),]
data_plot[c((iter_NR_EM_alpha_1[sel]+1):max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),2]<-data_plot[iter_NR_EM_alpha_1[sel],2]
```

Finally we create the Figure.

![](Cheating-Data_files/figure-markdown_github/unnamed-chunk-57-1.png)
