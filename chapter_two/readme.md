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


sidenote on stratified sampling:
    it means sampling with taking onto account the population bias
    if there are 51.3 male and 48.7 female
    then a stratified sample of 1000 will have 513 male and 487 female

5. visualising the data to gain insights
![](ocean.png)

6. Look for correlations
    
