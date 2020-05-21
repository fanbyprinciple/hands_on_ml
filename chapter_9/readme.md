## Tensorflow

Tensorflow makes computation easier by making a graph of computation and then running it using efficient C++ code.

More importantly youcan devide the graph for parallel GPU operation

Also supports distributed computing,it means it can be used over multiple servers 

Developed by google brain team that also created google photos and search

Most features are there in other libraries , it was open sourced.

With tensorflow we can also run quantum physics simulations if we wanted

tf.learn(not TFLearn) compatible with scikit-learn

tf-slim for simplifying building, training and evaluating neural network

provides advanced optimisation nodes to search for parameters to minimise a cost function. Tensorflow autmatically takes care of computing gradients of function we define

tensorboard to visualise computation graph, view learning graph.
Complex graphs yet to understand.

![](tensorboard_visualisation.png)

Example for modularity

![](modularity.png)

Need to do from modularity in book.

Namescopes : these are a way ot organise tensorboead graph
as similiar variables collapse into same namescope

Modularity : 
A Relu copmutes the linear function og inputs outpusts and result if it is positive ,0 otherwise
$h^w,b$(X) = max(X.w + b,0)

Exercises:

1. Main advantages of using a computation graph rather than using execution

    1. tensorflow automatically computes the gradients for you(using reverse mode autodff)
    2. tensorflow can take care of running operations in parralel threads
    3. makes it easier ti run the same code accross different devices
    4. It simplifies introspection
  Drawback:
      1. so hard to even understnad
2. a_val = a.eval(session=sess)
   and 
   a_val = sess.run(a)
   are equivalent

3. a_val,b_val = a.eval(session=sess),b.eval(session=sess)
    and a_val, b_val = sess.run([a,b]) arenot equivalent.
    frist statement run the graph twice. while second runs only once. any change in the run willbe reflected onthe other variable.
    
4. We can't run two graphs in same session

5. If we create two threads sharing same variable , in normal tensorflow operation then starting two threads and opening a local session in each of those threads, both using the same graph g, then each sesssion will have its own copy of the variable w. However if connected to same cluster then theyll sahre same w .used in distributed tensorflow.

6. A variable is initialized and destroyed when the session ends . Not true in distributed tensorflow.

7. Variables is an operation that holds a value. If yourun the variable, it returns the value, You need to initialise it . Youcan changethe variable's value(for example by using assignment operation)  it is stateful. It keepsits value in multiplr run
Variable can't be refreshed in between execution

Placeholders technically don'tdo much the just hold information about the type and shape of tensor they represent but have no value, we need to feed in value of feed_dcit to the placeholder.
Typically used to feed training ot test data in execution phase
Placeholders don'thold value they are just placeholders for values youll provide

8. Exception is raised,if placeholder dowan't get a value and an operation is expected out of it

9. running graph can be used to to be fed into the ouputvalue of any operation

10. Setting value during execution phase
we define a variable, then if we want to change taht variable value  we use tf.assign()
for ex :

import tendorflow as tf
x =tf.Variable(tf.rand_uniform(shape=(), minival=0.0, max_val=1.0))
x_new_val = tf.placeholder(shape=(), dtype=tf.float32)
x_assign = tf.assign(x, x_new_val)

with tf.Session():
    x.initializer.run()
    print(x.eval())
    x_assign.eval(feed_dict={x_new_val:5.0})
    print(x.eval())

11. Reverse-mode autodiff needs to traverse the graph twice to compute the gradients of cost function with regards to number of variable

Forward mdoe autodiff will need to run once for each varaible 

Symbolic diffrentiation it would build a different graph to compute the gradients  so it dowsn't traverse the old graph at all

12. Logistic regression

moon dataset;
![](log_reg1.png)

logistic regression on test data:
![](log_reg2.png)

logistic regression after adding x^2 and x^3 to training
![](log_reg3.png)











