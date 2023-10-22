# Results

This markdown file aims to store the best results we obtain with the models at the different phases of training. The first phase of training consists before the improvements with the added column of last month and the redesign of the preprocessing pipeline.

## First phase

### sgd_regressor et linear_regressor

With both models we obtain the following results:
- mse : 159.8
- rmse : 12.6
- r2 : 0.115

### decision_tree_reg

With parameters 7 for the maximal depth and 8 for the minimum samples per leafes, we obtain the following results:
- mse : 144.4
- rmse : 12.0
- r2 : 0.200

### random_forest

With parameters 100 for the number of estimators, 7 for the maximal depth and 8 for the minimum samples per leaf, we obtain the following results:
- mse : 144.0
- rmse : 12.0
- r2 : 0.202

## extra_tree_regressor

With parameters min_samples_split  = 130, n_estim = 160, min_samples_leaf = 1, max_depth = 10, we obtain the following results:
- mse = 143.0
- rmse = 11.9
- r2 = 0.207

## Second phase

### GBR

With parameters n_estim = 1000, max_depth = 5, learning_rate = 0.01, min_samples_split = 140, min_sample_leaf = 5, we obtain the following results:
- mse = 100.7
- rmse = 10.0
- r2 = 0.435

### HGBR

With parameters max_iter = 50, max_depth = 7, min_samples_leaf = 20, learning_rate = 0.1, we obtain the following results:
- mse = 101.07
- rmse = 10.05
- r2 = 0.433

### Extremely Random Tree

With parameters n_estim = 300, max_depth = 25, min_samples_split = 27, min_samples_leaf = 1, we obtain the following results:
- mse = 96.51
- rmse = 9.82
- r2 = 0.4588

### XGBRegressor

With parameters n_estimators = 100, learning_rate = 0.3, max_depth = 3, we obtain the following results:
- mse = 101.89
- rmse = 10.09
- r2 = 0.428

### decision tree

With parameters max_depth=7, min_samples_leaf=5, we obtain the following results:
- mse = 108.28
- rmse = 10.405
- r2 = 0.393

### Linear regression

We obtain the following results:
- mse = 109.7
- rmse = 10.5
- r2 = 0.39

### random forest

With parameters n_estim = 300, max_depth = 20, min_samples_leaf = 15, min_samples_split = 2, we obtain the following results:
- mse = 96.27
- rmse = 9.81
- r2 = 0.460

### Lasso regression

We obtain the following results:
- mse = 122.4
- rmse = 11.1
- r2 = 0.31

### Ridge regression

With parameters alpha = 1, max_iteration = 1000 and tolerance = 0.0001, we obtain the following results:
- mse = 109.7
- rmse = 10.5
- r2 = 0.39
