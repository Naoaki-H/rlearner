#' Gradient boosting for regression and classification with cross validation to search for hyper-parameters (implemented with xgboost)
#'
#' @param x the input features
#' @param y the observed response (real valued)
#' @param weights weights for input if doing weighted regression/classification. If set to NULL, no weights are used
#' @param k_folds number of folds used in cross validation
#' @param objective choose from either "reg:squarederror" for regression or "binary:logistic" for logistic regression
#' @param ntrees_max the maximum number of trees to grow for xgboost
#' @param num_search_rounds the number of random sampling of hyperparameter combinations for cross validating on xgboost trees
#' @param print_every_n the number of iterations (in each iteration, a tree is grown) by which the code prints out information
#' @param early_stopping_rounds the number of rounds the test error stops decreasing by which the cross validation in finding the optimal number of trees stops
#' @param nthread the number of threads to use. The default is NULL, which uses all available threads. Note that this does not apply to using bayesian optimization to search for hyperparameters.
#' @param verbose boolean; whether to print statistic
#'
#' @return a cvboost object
#'
#' @examples
#' \dontrun{
#' n = 100; p = 10
#'
#' x = matrix(rnorm(n*p), n, p)
#' y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)
#'
#' fit = cvboost(x, y, objective="reg:squarederror")
#' est = predict(fit, x)
#' }
#'
#' @export
cvboost = function(x,
                   y,
                   weights=NULL,
                   k_folds=NULL,
                   objective=c("reg:squarederror", "binary:logistic"),
                   ntrees_max=1000,
                   num_search_rounds=10,
                   print_every_n=100,
                   early_stopping_rounds=10,
                   nthread=NULL,
                   verbose=FALSE){

  objective = match.arg(objective)
  if (objective == "reg:squarederror") {
    eval = "rmse"
  }
  else if (objective == "binary:logistic") {
    eval = "logloss"
  }
  else {
    stop("objective not defined.")
  }

  if (is.null(k_folds)) {
    k_folds = floor(max(3, min(10,length(y)/4)))
  }
  if (is.null(weights)) {
    weights = rep(1, length(y))
  }

  dtrain <- xgboost::xgb.DMatrix(data = x, label = y, weight = weights)

  best_param = list()
  best_seednumber = 1234
  best_loss = Inf

  if (is.null(nthread)){
    nthread = parallel::detectCores()
  }

    for (iter in 1:num_search_rounds) {
      param <- list(objective = objective,
                    eval_metric = eval,
                    subsample = sample(c(0.5, 0.75, 1), 1),
                    colsample_bytree = sample(c(0.6, 0.8, 1), 1),
                    eta = sample(c(5e-3, 1e-2, 0.015, 0.025, 5e-2, 8e-2, 1e-1, 2e-1), 1),
                    max_depth = sample(c(3:20), 1),
                    gamma = runif(1, 0.0, 0.2),
                    min_child_weight = sample(1:20, 1),
                    max_delta_step = sample(1:10, 1))

      seed_number = sample.int(100000, 1)[[1]]
      set.seed(seed_number)
      xgb_cv_args = list(data = dtrain,
                         param = param,
                         missing = NA,
                         nfold = k_folds,
                         prediction = TRUE,
                         early_stopping_rounds = early_stopping_rounds,
                         maximize = FALSE,
                         nrounds = ntrees_max,
                         print_every_n = print_every_n,
                         verbose = verbose,
                         nthread = nthread,
                         callbacks = list(xgboost::cb.cv.predict(save_models = TRUE)))

      xgb_cvfit <- do.call(xgboost::xgb.cv, xgb_cv_args)

      metric = paste('test_', eval, '_mean', sep='')

      min_loss = min(xgb_cvfit$evaluation_log[, metric])

      if (min_loss < best_loss) {
        best_loss = min_loss
        best_seednumber = seed_number
        best_param = param
        best_xgb_cvfit = xgb_cvfit
      }
    }

  set.seed(best_seednumber)

  xgb_train_args = list(data = dtrain,
                        params = best_param,
                        nthread = nthread,
                        nrounds = best_xgb_cvfit$best_ntreelimit)

  xgb_fit <- do.call(xgboost::xgb.train, xgb_train_args)

  ret = list(xgb_fit = xgb_fit,
             best_xgb_cvfit = best_xgb_cvfit,
             best_seednumber = best_seednumber,
             best_param = best_param,
             best_loss = best_loss,
             best_ntreelimit = best_xgb_cvfit$best_ntreelimit)

  class(ret) <- "cvboost"
  ret
}

#' Gradient boosting for regression and classification with cross validation to search for hyper-parameters (implemented with xgboost)
#'
#' @param x the input features
#' @param y the observed response (real valued)
#' @param weights weights for input if doing weighted regression/classification. If set to NULL, no weights are used
#' @param k_folds number of folds used in cross validation
#' @param objective choose from either "reg:squarederror" for regression or "binary:logistic" for logistic regression
#' @param ntrees_max the maximum number of trees to grow for xgboost
#' @param num_search_rounds the number of random sampling of hyperparameter combinations for cross validating on xgboost trees
#' @param print_every_n the number of iterations (in each iteration, a tree is grown) by which the code prints out information
#' @param early_stopping_rounds the number of rounds the test error stops decreasing by which the cross validation in finding the optimal number of trees stops
#' @param nthread the number of threads to use. The default is NULL, which uses all available threads. Note that this does not apply to using bayesian optimization to search for hyperparameters.
#' @param verbose boolean; whether to print statistic
#'
#' @return a cvboost object
#' @export
cvlgb <- function(x,
                  y,
                  weights=NULL,
                  k_folds=NULL,
                  objective=c("reg:squarederror", "binary:logistic"),
                  ntrees_max=1000,
                  num_search_rounds=10,
                  print_every_n=100,
                  early_stopping_rounds=10,
                  nthread=NULL,
                  verbose=FALSE) {
  objective = match.arg(objective)
  if (objective == "reg:squarederror") {
    objective = "regression"
  }
  else if (objective == "binary:logistic") {
    objective = "binary"
  }
  else {
    stop("objective not defined.")
  }

  if (verbose) {
    verbose = 1L
  }
  else {
    verbose = -1L
  }

  if (is.null(k_folds)) {
    k_folds = floor(max(3, min(10,length(y)/4)))
  }
  if (is.null(weights)) {
    weights = rep(1, length(y))
  }

  dtrain <- lightgbm::lgb.Dataset(
    data = as.matrix(x),
    label = y,
    weight = weights
  )

  best_param = list()
  best_seednumber = 1234
  best_loss = Inf

  if (is.null(nthread)){
    nthread = parallel::detectCores()
  }

  for (iter in 1:num_search_rounds) {
    param <- list(objective = objective,
                  bagging_fraction = sample(c(0.5, 0.75, 1), 1),
                  feature_fraction = sample(c(0.6, 0.8, 1), 1),
                  learning_rate = sample(c(5e-3, 1e-2, 0.015, 0.025, 5e-2, 8e-2, 1e-1, 2e-1), 1),
                  max_depth = sample(c(3:15), 1),
                  lambda_l2 = runif(1, 0.0, 0.2),
                  min_child_weight = sample(1:20, 1),
                  max_delta_step = sample(0:10, 1),
                  nthread = nthread
                  )

    seed_number = sample.int(100000, 1)[[1]]
    set.seed(seed_number)
    lgb_cv_args = list(data = dtrain,
                       param = param,
                       nrounds = ntrees_max,
                       verbose = verbose,
                       eval_freq = 0L,
                       early_stopping_rounds = early_stopping_rounds,
                       init_model = NULL,
                       callbacks = list(),
                       nfold = k_folds)


    lgb_cvfit <- do.call(lightgbm::lgb.cv, lgb_cv_args)

    min_loss = lgb_cvfit$best_score

    if (min_loss < best_loss) {
      best_loss = min_loss
      best_seednumber = seed_number
      best_param = param
      best_lgb_cvfit = lgb_cvfit
    }
  }

  set.seed(best_seednumber)

  lgb_train_args = list(data = dtrain,
                        params = best_param,
                        verbose = -1,
                        eval_freq = -1,
                        nrounds = best_lgb_cvfit$best_iter)

  lgb_fit <- do.call(lightgbm::lgb.train, lgb_train_args)

  ret = list(lgb_fit = lgb_fit,
             best_lgb_cvfit = best_lgb_cvfit,
             best_seednumber = best_seednumber,
             best_param = best_param,
             best_loss = best_loss,
             best_ntreelimit = best_lgb_cvfit$best_ntreelimit)

  class(ret) <- "cvboost"
  ret

}

#' predict for cvboost
#'
#' @param object a cvboost object
#' @param newx covariate matrix to make predictions on. If null, return the predictions on the training data
#' @param ... additional arguments (currently not used)
#'
#' @examples
#' \dontrun{
#' n = 100; p = 10
#'
#' x = matrix(rnorm(n*p), n, p)
#' y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)
#'
#' fit = cvboost(x, y, objective="reg:squarederror")
#' est = predict(fit, x)
#' }
#'
#' @return vector of predictions
#' @export
predict.cvboost <- function(object,
                            newx=NULL,
                            ...) {
  if (is.null(newx)) {
    return(object$best_xgb_cvfit$pred)
  }
  else{
    dtest <- xgboost::xgb.DMatrix(data=newx)
    return(predict(object$xgb_fit, newdata=dtest))
  }
}
