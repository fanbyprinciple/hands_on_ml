## Knowing the task to classify

multivariate regression - multiple features are taken into consideration
univariate regression - single feature is considered

If the data is huge then we can do online learning, maybe even implement mapreduce.

1. Know the task
2. Select a performance measure

    Ex - Root mean square error (RMSE)
    t measures the standard deviation of the errors the system makes in its   predictions

    rmse(X,h) = root of ((sum of h(x) - y/ mean) ^ 2)

    A small root mean square program in python :
    ![](rms.png)

3. Check the assumptions

The histogram after loading data

![](histo.png)
