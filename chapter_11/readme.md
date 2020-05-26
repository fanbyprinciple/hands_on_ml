# Deep Neural nets

Problems with training a DNN:

    1. Vanishing gradients problem : gradients vanish at lower layer
    2. Larger thenetwork slower it is to train
    3. A model with a million parameters can overfit the network

### Vanishing gradients

gradients get smaller and smaller as algorithm reaches loewr layers seen with function with logistic activation
exploding gradients where the gradient become very big.

![](exploding_gradients.png)

2010 paper Understanding the difficulty of training deep feedforward network

Xavier and He solution to disappearing gradients problem
1. We need to ensure that variance of inputs is equal to the variance of the outputs
2. Gradients should have equal variance before and after flowing through a layer in the reverse direction
3. For this to happen the input and output initialisations must be the same

The proposed a Xavier initialisation or glorot initialisation where weights are initilaised randomly.

fully_connected() function uses xavier initialisation, which can be chagned to a He initiliser

He initilisationconsiders only fan in not the average betweenfan nin and fan out. This is the default for variable_scaling_initialiser() function

### Non saturating activation function

ReLu was much better tahn sigmoid activation because:

1. It is faster to compute
2. It doesn't saturate for positive values

But they also suffer from dying relu problem: 
    some neuron only output 0
so we introduce a concept of leaky relu leaky reulu is F(x) =  max(alpha *z, z) 
The hyperparamter alpha defines how much the function "leaks": it is the slope ofthe function for z < 0 and is typiclaly set at 0.01
This small slope ensures that they never die. THey goto a deep coma but they
never die. They have a chance to eventually wake up.

![](leaky_relu.png)
In fact setting alpha = 0.2 performed better than samll leak of 0.01

Randomised leaky relu (RReLU) where alpha is picked randomly in a given range and fixed to a aerage value during training. 

Parametric leaky ReLU (PReLU) where alpha is authorised to be learned during training (instead of being a hyperparameter, it becomes a parameter). It outperforms the smaller.

ELU - exponentail Linear unit ouperformed Relu
![](elu.png)

1. It takes on negative values when z < 0 which allows the output to be closer to 0
It helps alleviate the vanishing gradients. lapha is usaully set at 1

2. It has non zero gradient for z<0 thus dyunig units issue is avoided

3. It is smooth everywhere including z<0

The manideawback is it is slower to compute bbut durin gtraining it has faster convergence rate. at test time elu is lower

for deep neural networks:
ELU > leaky RElu > Relu > tanh > logistic

There is something called SELU which is even better.

![](selu.png)

### Batch Normlisation

It is given as a solution to exploding graidents.
In this we add an operation tomoel just before ativation function of each layer,simpy centring and noramlising the inputs, then scaling andshifting the result. It lets you find the optimal scale and mean of inputs of each layers.

inorder to zero center and normalise the algorithm needs to estimate the mean and standard devaition. I evaluated the mean and standard devaition
At test timewe use the entire mean and standard devaition since mini batch is not available

Batch normailisaation removes the need because it add complexity but noramlisation is not needed
