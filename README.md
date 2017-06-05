# nEM
This repository refers to the paper  [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), where we propose a novel **nested EM** for improved maximum likelihood estimation of **latent class models with covariates**. The proposed **nested EM**  relies on a sequence of conditional expectation-maximizations which leverage the recently developed [Pòlya-Gamma data augmentation](http://www.tandfonline.com/doi/abs/10.1080/01621459.2013.829001) for logistic regression to obtain simple M-step via  generalized least squares. As we discuss in the paper, differently from current algorithms, the proposed **nested EM** provides a monotone log-likelihood sequence, and allows substantial improvements in maximization performance, according to empirical studies.

## Source code

The repository contains the source functions `LCA-Covariates-Algorithms.R` for the implementation of the proposed **nested EM** (and an hybrid version of the **nested EM**), along with additional algorithms routinely considered in the estimation of **latent class models with covariates**. 

## Empirical performance assessments

As we carefully show in Sections 3.1 and 3.2 of the paper [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), the **nested EM** provides substantial improvements in maximization performance, compared to current algorithms.
