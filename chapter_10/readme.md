# Artificial Neural Network

Artificial neural netwrok werer first introduced backin 1943 by Warren McCulloh on propositional logic.

Perceptron introduced in 1957
the decision boundary of each output neuron is linear so perceptron are incapable of learning complex patterns, just like logistic regression calssifiers
This is also called the perceptron convergence theoram

scikit learn provides a Perceptron class, that implements a single LTU network.

![](perceptron_iris.png)

Since perceptron couldn't solve XOR problemthey had dropped it altogether

However it turns out that thel limitation of Perceptron can be solved by stacking multiple perceptron

![](mlp_xor.png)

Neuron that fire together wire together

### Multi layer perceptron and back propagation

A MLP is composed of one(passthrough) input layer and followed by one of more layers of LTU, and one final layer called the otput layer

When a ann has two ormore hidden layers it is called a deep neural network

in 1986, backpropagation was introduced. It can be described as Gradient Descent autodiff.

Gradient descent is the algorithm that involves updating a set of parameters to minimize a loss, and is typically in the form of

θt+1=θt−α∇θJ 

The nabla (upside down triangle) is the gradient of the loss with respect to the parameters, theta. To find this gradient, you need to differentiate the loss with respect to the parameters.

Automatic differentiation is a way of automating the acquisition of these derivatives. Normally, you would work out tediously the derivatives of each of the operations, and find how they combine to get the derivative of the loss wrt each of the parameters, but with automatic differentiation, this is algorithmically generated. The found derivatives are typically slower than hand crafted derivatives since humans are able to optimize unnecessary steps, which computers may not be able to find, but automatic differentiation way outweighs the disadvantages.

for each training instance, algorithm feeds it to the network and computes the output of every neuron in each consecutive layer, (this is the forward pass, just likewhen making predictions) then it measure the output error, and then computes how much each neuron in last hidden layer contruted to each ouput neuron's error. it continues till it reaches the input layer. 
forward and reverse passes of backpropatgation is simply reverse autodiff. The last step is a gradient descent based on the error calculated all accross the network

Or to put it simply. for each training instance the backpropagation algo‐
rithm first makes a prediction (forward pass), measures the error, then goes through
each layer in reverse to measure the error contribution from each connection (reverse
pass), and finally slightly tweaks the connection weights to reduce the error (Gradient
Descent step). 

### some of the activation functions
For gradient descent to work properly we replace the step function with logistic function sigma(z) = 1/(1 + exp(-z))

other popular activation functions are -
hyperbolic tangent function tanh(z) = 2sigma(2*z) - 1
the value is between -1 and 1, which helps in sppeding up to algoritm

Relu activation function -
Relu(z) = max(0,z) it is continuous bu not diffrentiable at z = 0, however it works fast

Activation functions:
![](activations.png)

Their derivative:
![](activations_derivative.png)

### Feed forward Neural network
The ouput of each neuron corresponds to the estimated probability of the corresponding class. Notice that the signal only flows in one direction from inputs to outputs.

### Using plain tensorflow

Using the number of hidden layer to be at 300 and 100, creating a placeholdervariable fo reach of the training instances. The actual nerual netweor has 2 hidden layers and they only differ by the number of inputs and the outputs they contain, output uses softmax












