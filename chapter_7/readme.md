# Ensemble earning

1. Works on the principle of wisdom of the crowd.

an example where a small bias can go a long way when number of tosses are more:
![](toss.png)

2. We combine multiple classifiers. And take the classifier with the best predictions. Even if the classifiers are weak still the combined aggregate can be a strong learner.

![](simple_ensemble.png)

### Bagging and pasting

1. we can train different algorithms oruse the same algorithmon differnet subsets of datatsets. WHEN SAMPLING IS PERFORMED WITH REPLACEMENT IT IS CALLED BAGGING AND WIHTOUT REPLACEMENT IT IS CALLED PASTING.

In other words both bagging and pasting allow the training instances to be sampled several times accross multiple predictor but only bagging allows the training instances to be sampled several times forthe same predictor.

We use the statistic mode for aggregating the final decision.Aggregatingreduces both bias and variance, the prediction can be done parallely.

![](bagging.png)

bagging classifier automatically does soft voting if the base classifier can estimate class probabilities (if it has a predict_proba function), bagging often results in better performance than pasting.

### Out-of-bag evaluation

With bagging some instances may be sampled several times for any given predictor, while others may not be sampled at all. By default a bagging calssifier samples mtraining instances with replacement (bootstrap=True) where m is the size of the training set.It means only 67 percent of all training instances are sampled rest 33 are not sampled. The training samples that are not sampled are called outof bag oob instances.

This oob can be used for cross validation.

This can be done automatically in BaggingClassifier

### Random patches and Random subspaces

BaggingClassifier supports sampling of features as well through max_features and bootstrap_features. They work the same way as max_samples and bootstrap. but for feature sampling than instance.

This is useful when dealing with high dimensional input such as images. sampling both training instances and images is known as random patches method. keeping training instances (bootstrap=False and max_Samples=1.0) and sampling features (bootstrap=true and or max_features smaller than 1.0 is called random subspaces method.

Sampling features results in even more predictor diversity trading a bit more bias for lower variance. 

### Random forests

Random forest is an ensemble of Decision Trees, generally trained via bagging method (or sometimes pasting) and max_samples are set to the size of training set.

Instead of building a BaggingClassifier and passing it a DecisionTreeClassifier you instead use RandomForestClassifier which is optimised for decison class using available CPUcores

Random forest algorithm introduces extra randomness when growing trees instead of searching for the very bet feature when splitting a node. This results in greater tree diversity. 

![](random_forest_moon.png)

### Extra trees

when you are growing a tree in Random forest at each node only a random subset of features is considered for splitting. It is possible to make trees even more random by also using random thresholds for each feature than searching for the best possible thresholds (like regular Decision Trees)

A forest of such extremely random trees is simply called an Extremely Randomized Trees ensemble. This adds more bias and less variance.

You can create an Extra-trees calssifier using Scikit learn ExtraTreesClassifier class. SImilarly ExtraTreesRegeressor class also exists.

### Feature importance

important features appear at root of tree while unimprtant features often appear closer to leaves or not at all.

![](feature_importance.png)

### Boosting

Boosting originally called hypothesis boosting refers to an enseble meathod tha combine weak learners to make a strong learner. 

Trains predictors sequentially, each trying to correct its predecessor.

### Adaboost

IN adaboost the new predictor pays a bit more attention to the training instances that hte predecessor underfitted. 

![](adaboost_moon.png)

The wrong classifications weights are boosted in a sequentia manner, so instead of minismising a cist function adaBoost adds predictors to the ensemble gradually making it better.

![](adaboost_learning_rate.png)

adaboost uses a multiclass version denoted by SAMME (stagewise additive modelling using a multiclass exponential function)
If there are two classes adaboost is equivalent to samme.If predicts_proba() function can beused then the SAMME performs better.

WE trained Based on 200 decision strumps. using AdaboostClassifier . A decision stump is  adecision tree with max depth 1.

### Gradient boosting

In gradient boosting, it tries to fit new predictor tothe residual errors made by previous predictor. Regression is done by gradient tree boosting.  To finally make a regression we just add all the predictions.

![](gradient_boosting.png)

We can use GradientBoostRegressor class to create ensemebles. much like the random forest regressor class.It can control the growth of Decision trees (eg max_depth, min_samples_leaf) and hyperparameters to control training, such as the no. of trees n_estimators.

learning_rate decides the contribution of each tree. If low then it generalisaes better. also called shrinkage, used as a regularisation technique. as an example off too few and too many tree trees affect on predictions.

![](gbrt.png)

We can use early stopping through staged predict() method,return an iterator over the predictions made by the ensemble at each stage of training. The following code trains a GBRT ensemble with 120 trees then measures the validation error at each stage of training. and finally trains with an optimum number of trees.

![](optimum_tree.png)

Subsample hyperparameter in GradientBoostingRegressorclass specifies the fraction of training instances to be used for training each teree. This trades higher bias for a low variance. It speeds up the training. This is called Stochastic Gradient Boosting.

### Stacking

Stacked generalisationn. Instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble.  So the model that chooses the predictions between the classifier is also trained which one to choose.also called a meta learner or blender.

It is possible to train different blenders so you can stack blender on top of another. 

Scikit doesn't support stacking automatically but we can perform out own implementation.

## Exercises
1. If you have trained five different models on the exact same training data, and
they all achieve 95% precision, is there any chance that you can combine these
models to get better results? If so, how? If not, why?

we can.It will work if the modelsare very different , it will work better if models are trained on different training data.

2. What is the difference between hard and spft voting ?

Hard voting counts the votes and picks the one with most votes
Soft voting computes the average estimated class probability of each class and then picks the best class. Soft margin only works if it has predict_proba function,it gives high confidence votesmore weight.

3. Can we speed bagging ensemble by distributing it over multiple servers. what about pasting ensembles , boosting ensembles, eandom  forest, stacking ensembles

Since each predictor in the ensemble is independent of the others. It ccan be done of pasting ensemebles and random forest and bagging ensembles. However in a boosting ensembles we cannot do that because it depends on previous predictions. It can be dine in stacking at layer level only after previous layer is done.

4. wah tis the benefit of out of bag evaluation ?

makes use of features not selected in bagging for validation

5. What makes extra trees more random than regular forest ?

In random forest only a random subset of features is considered for splittinga  node. however in extra trees, they go one step further, rather than searching for ht best possible thresholds they use random thresholds for each feature. This acts like regularisation.Inpractice they are not much higher or lower when making predictions

6. WE can increase the number of estimators so to avoid underfutting  in Adaboost we can also increase the learning rate.

7. If gradient boosting overfits we should reduce the learning rate. we can also use early stopping

8. Voting classifier on mnist
![](voting_classifier_on_mnist)

9. Stacking clssifier on mnist
![](stacking_classifer_on_mnist)


