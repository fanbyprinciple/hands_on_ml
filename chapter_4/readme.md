Training a model

    Training a model loosley translates to setting its parameters so that the model best fits the training set. For this we need to see what is the effect of different parameters on training data
    
    The most common performance measure is RMSE. Therefore,
    Training a linear model means we need to find the value of theta that minimises RMSE
    In practice it is simpler to minimise the Mean Square Error (MSE) than RMSE. Which also gives us the same result.
    
Linear regression
    
    finding parameters using normal equationfor mean square error
    
    ![](lin_reg.png)
    
    Using scikitlearn default,
    
    ![](lin_reg_1.png)
    
Computational complexity

    The normal equation gets very slow when the number of features grows large (e.g. 100,000).
    Computation complexity of inverting a X.T.dot(X) is O(n^3)
    However noraml euqtion is linear with regards to the number of instances in the training set so that it can handle large training sets efficiently.
    And once the model is trained the predictions are very fast
    the computational requirement is linear

Different ways to train a linear model

    Gradient Descent : Tweaks parameters iteratively to minimize the cost function
    It measures the local gradient of errorfunction with regards parameter vector theta, and goes in the direction of descending descent. Once the gradient is zero, you have reached the minimum.
    
    In essesnce you'll fill theta with random values and then improve it gradually throughout the training.
    The size of steps is learning rate, youll have to go though many iterations to converge.
    
    if its too small then it takes a long time
    if its too big then it makes algo diverge
    
    it can also reach local minimum instead of global minimum
    
    The cost function of MSE happens to be a convex function, which means that if you pick any two points on the curve the line swgment joining them will never cross the curve it means 
    theres is no local minima only a global minimum
    its a continuos function that never changes abruptly
    
    We need to ensure that all features have similar scale(range) using StandardScalar
    
    Training a model in this context means a search in models parameter space, more parameter more dimesnions means more time to search
    