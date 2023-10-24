# Parametric Fairness

This is the code repository for our paper, with some additional details on the simulations. 

# Simulations

## Folktables

The data is available through the `folktables` package see [github.com/socialfoundations/folktables](https://github.com/socialfoundations/folktables)

### Regression

Based on the ACSIncome task, but here the target variable was not binarised. Instead, we use the same explanatory variables as in the standard application and want to predict the log of PINCP. The data was filtered according to the following cutoffs: at least 18, at least one hour worked, at least 100 in income.

To simulate corrupted data, we added noise from a Gamma(1,0.5) distribution to training observations drawn with the help of a Bernoulli. The p parameter of the Bernoulli is then subsequently changed (0,0.25,0.50,0.75) indicating a stronger bias in the data. The test set remained unchanged. The randomness in the experiments comes from the seed used to split the dataset into train/test, which was done with a 80%/20% ratio. Since the goal of the application is to illustrate how different post-processing methods affect the outputs we only spent a minimal time on the base models themselves. We used a LightGBM base models, where we limited the minimum number of points per terminal node to 50, and added column sampling and then estimated the number of necessary boosting iterations via five-fold cross-validation. The data was then retrained on the whole training set with the number of iterations found via CV (to avoid overfitting). Here the performance metric is measured by MSE. 


### Classification

Based on ACSPublicCoverage Task from the package. Here we added the sensitive feature based on the PINCP column. The observed sensitive feature was 1 if the income was lower than 45'000. The unobserved feature was 1 if the income was lower than 15'000. We opted for this as income is often only reported in brackets and hence such a situation could arise in real situations. Note that the filter criteria were increased to 60'000 as opposed to the standard 30'000 to suit the new application. 

The randomness in the experiments comes from the seed used to split the dataset into train/test, which was done with a 80%/20% ratio. Since the goal of the application is to illustrate how different post-processing methods affect the outputs we only spent a minimal time on the base models themselves. We used a LightGBM base models, where we limited the minimum number of points per terminal node to 50, and added column sampling and then estimated the number of necessary boosting iterations via three-fold cross-validation. The data was then retrained on the whole training set with the number of iterations found via CV (to avoid overfitting). The F1 score needs to have a cutoff parameter, which was chosen by optimising the value on the train set and then fixing it for all experiments. 

## Compas

The data is available on probulica's github repository [github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis). We used the `compas-scores-two-years.csv` dataset form the master branch and the simulations expect it to lie in a folder at `./data/compas/`. The task is to predict `is_violent_recid`, with the observed sensitive variable being middle aged and the latent sensitive variable being aged 45 or over. This was chosen as the original investigation [propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) showed that those variables have a high influence in a basic model. 

The randomness in the experiments comes from the seed used to split the dataset into train/test, which was done with a 80%/20% ratio. Since the goal of the application is to illustrate how different post-processing methods affect the outputs we only spent a minimal time on the base models themselves. We used a LightGBM base models, where we limited the minimum number of points per terminal node to 20 and simply estimated the number of necessary boosting iterations via three-fold cross-validation. The data was then retrained on the whole training set with the number of iterations found via CV (to avoid overfitting). The F1 score needs to have a cutoff parameter, which was chosen by optimising the value on the train set and then fixing it for all experiments. 


# Auxiliary

## Visualisations

The Figures can entirely be reproduced by running the notebook `./notebooks_application/visualisation_figs.ipynb`. Note that you might need to create the parent folder (`./data/example_data/`)



## Version and Hardware

Follow the `requirements.txt` file for details on versions of specific packages. Python version used throughout the experiments was `3.10.10` and experiments were run on MacBook Pro M1 2020