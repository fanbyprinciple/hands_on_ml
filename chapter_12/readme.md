The chpater covers:

1. Muliple GPU devices
2. setting up and starting a tensorflow cluster
3. Distributing computations accross multiple devies and server
4. Sharing variables accrosssessions usnig containers
5. Coordinating variablesusing queues and readers 
6. Coordinating multiple graphs working asynchronously using queues
7. Reading inputs efiiciently using readers, queue runners and coordinators
8. Parallelizing Neural Network on a tensorfow cluster

1. run `nvidia-smi` on command line too get details about our GPU.


### Managing GPU RAM

1.You can run programs by specifing GPU utilisation
    1. CUDA_VISIBLE_DEVICES=0,1 python3 program_1.py
    2. CUDA_VISIBLE_DEVICES=3,2 python3 program_2.py

2. We can use config = tf.configProto() to tell how much of memory to utilize by the gpus.

3. Yet another option is to tell tensorflow to grab memory only when it requires it
we set `config.gpu_options.allow_growth` to true.

### Placing operations on Devices

1. Simple placement

TensorFlow uses simple placer to place it.
    1. if a node was already placed on a device in previous run it is left on the device
    2. If user pinned a node to a device, the placer places it on the device
    3. Defaults to GPU 0
   
To pin nodes onto a device you must create a device block using the device() function
using `with tf.device("./cpu:0")`

2. Dynamic placemnt

When creating a device block we can specify a function instead of a device name
Tensordlow will call this function for each operation it needs to place in the device block, and the function must return the device to pin the operation on


3. Operations and kernels

when tensorflow tries to run on a device it will have an implementation for that device, called a kernel. Many operations have kernels for both CPUs and GPUs but not all of them. FOr example tensorflow dows not have a GPU kernel for integer variables. you have to make it float.

4. soft placement

By default if you try to pon an operation on a device for which the operation has no kernel, we get the exeption however if we want the execution to falll back on cpu instead we can use `allow_soft_placement` configuration option to True.

### Parallel Execution

When Tensorflowruns a graph it starts by finding the list of nodes that need to be evaluated and it counts how many dependecies each of them has. Tensorflow then evalutest the node with zero dependencies.
If nodes are placed on seperate devices they obviously get evaluated in parallel too(in separate GPU threads and CPU cores)

TF maintatins a thread poolon each device to parallelizeoperations.They are called inter op thread pools. These operations have multi threaded kernels They cn use other threadpools one per device called intra op thread pools.

`inter_op_parallelism_threads` operatons for inter-op pool threads
`intra_op_parallelism_threads` operations for intra op pool threads

### Control Dependencies

Sometimes we might like to make some operations defer their occupation of ram
TOpostpone evaluationof somemodes we use control dependencies

### Multiple Devices Across Multiple Servers

To run a graph accross multiple servers you define a cluster. A is composed of one or more Tensorflow servers called tasks, typically spread accross several machines.

Each task belongs to a job ( a group of tasks)

example a cluster of "ps" parameter server
performing computation such as "worker". we cn pin it by first decalringa cluster spec then using tf.train.Server(cluster_spec, job_name="worker", task_index=0)

In order to ensure that they don't grab all ram we can
1. tell them to use specific gpusallocated
2. tell them to  use a fraction of gpu associated

If we want the process to do nothong other than tensorflow server you can block main threadby telling it to wait for the server as soon as your main exits. using `server.join()`

### Opening a Session

Once all the tasks are up and running , you can opwn a session on any of the servers, and use that asa regular local session

`with tf.Session("grpc://machine-b.example.com:2222") as sess`

The server will be the master to client that wants to compute the graph.

### Master and worker services

The client uses the gRPC protocol (google remote procedure call) to communicate with the server. This is an efficient open source framewoek to callremote functions and get their outputs across a variety of platforma nd languages. Based on HTTP2 which enables bidirectional binding.

Data is transmitted in theform of protocol buffers, it is a lightweight data interchange format.

appropriate ports shoulld be open

master service allows to open sessions and use them to run graphs

one client can connect to multiple servers.

### pining operations across tasks

You can use device blocks to pin operations onany device managed by any task, by specifying the job name,task index device type and device index. 

for example `with tf.device("/job:ps/task:0/cpu:0)"`
 if we omit this then we will run default to the local machine
 
### Sharding Variables across multiple parameter servers

we can store the modelparameters on a set of parameter servers( i.e the tasks in "ps" job) while other tasks focus on computations (i.e tasks in the "worker"" job). For large models with millions of parameters it is useful to shard the parameter across multiple parameter servers 

That is why we have replica_device_setter_to automatically pin tasks 
in a round robin fashion

`with device(tf.train.replica_device_setter(ps_tasks=2)`

inner block can always override the taks job or defined in outer block

### Sharing state across sessions using resource containers

When we are using a plain local session (no the distributed kind) each variables state is managed by session itself. as soons as it ends all variables values are lost.

Multiple local sessions cannot share any state, even if bothe run the same graph, each session has its own copy of every variable (in contrast when we use distributed session variable state variable is managed by resource conntainers located in the cluster itself not the sessions)

So for exampleif we have a variable named x using one client session it will automaticlly be availabel to any other session on the same cluster

Lets supppose you have a tensorflow cluster up and running on machines A and B port 2222. we cantekk the client to opena session at machine a by 
 `python simple_client.py grpc://machine-b.example.com:2222`

We can use variable scopes to avaoid clashes

### Asynchronous communication using tensprflow queues

Queues are another great way to exhange data between multiple sessions for example one common use case is to have a client create a graph that loads the training data and pushes it into a queue, while another client creates a garph that pulls the data from the queue and trains a model. This has the effect ofspeeding up communications

tensorflow queues use shared name to identify resources

To push data to a queue, we must create an enqueue operation 
To pull we use a dequeue operation

We can also provide the queue with a tuple of different tensors.When we dequeue and run these operations it is necessary that we run both at the same time otherwise they might be lost

evenn after close the queeu the pending items will be honoured by dequeue

RansomShufflequeue - they are sae as FIFOQueue except that the items are in random order

we can use dequeue_many(5) function to take ou multiple items from dequeue
in this instance 5 at a time

### Padding Fifo queue

A paddingFiFo queue can also be used just like a FIFOQueue except that it accepts tensors of variable sizes along any dimension (but with fixed rank) when we dequeue them a dequeue_many or dequeue_up_to operation each tensor is padded with zeros along every variable dimension to make it the same size as the largest tensor in the minibaatch

q = tf.PaddingFIFOqueue()

if we dequeue one item ata time we getexact same tensors that were enqueued, however when we use dequeue many we get padded response

This type of queue can beuseful when you are dealing with variable length inputs such as sequence of words

### Loading Data directly from the Graph

So far we assumed that lcients would load the training data and feed it to the cluster using placeholder

other methods are:
1. Preloading the data into a variable

For datasets that can fit in memory,a better option is to load the training data once and assign it to a variable then just use that variable
This is called preloading of the training set. 

We must also set trainable to false so that optimiser don't try and tweak this variable
and also set the collections=[] to ensure that it doesn'tget added to Graph Keys.GLOBAL_VARIABLES collection, this is used for saving and resotring checkpoints

This example assumes that all of your training sets consist of only float32 values

2. Reading the training data directly from the graph

We can use reader operations that will directl read data from filesystem. This waythe training data never needs to flow through the clients at all, tensorflow provides for various file formats

CSV
Fixed length binary records
TFown tfrecords format based on protocol buffers

we use a textlinereader to read this file. This will be output to a keyand value pair that can then be preprocessed and be done with rest of the workflow.

3. We can also multithread readers using a coordinator and QueueRunner

A coordinator is a very simple object whose sole purpose is to coordinate stopping multiple threads 
we create a cordinator then give it all threads that needs to stop jointlu . Any thread that can request that every thread stop by callingthe coordinators requeat_stop() method

A QueueRunner runs multiple threads that each run an enqueu operation repeatedly filling up a queue as fast as possible. As soon as the queue is closed the next thread that tries to push an item to the queue will get an Out ofRange Error this thread catches the error and immediately tells other threads to stop using a coordinator

The first line creates the queueTunner and tells it to run five threads, all running the same enqueue_instance operation repeatedly. Then we start a session and we enqueue the name of the files to read.Next we create a coordinato r taht queueRunner will use to stop gracefully

This will be a bit moreefficient than earlier but we can do better
we can read themfrom different files

4. Other convenient functions

string_input_producer()

shuffle_batch() takes a list of tensors and creates a RandomShuffleQueue, a queue runner a dequeue operation to extract mini batch

## Parallelizing neural network on a tensorflow cluster

Trickier problem of training a single neural network accross multiple devices and servers

### One neural network per device
Best way to train and neural network on  atensorflow cluster s to take theexact same code you would use for a single device on a single machine and specify masterserver address. we can change the device that willrun our graph by simply putting our codes construction phase withina device block.

This is perfect for hyperparameter tuning: each dvice in the cluster will train a different model with its own set of hyperparamers

This also words well if we host a web service that recieves a large number of queries per second(QPS) and we need our neural network to make a prediction for each query, We simply replicate the neural network accross all devices on the cluster and dispatch queries accross all devices.By adding more servers we can handle an unlimited number of QPS (This will not reduce time to process a single request howver)

There is also the use `tensorflow serving`
desgned to serve high amount of queries to machine learning models (typically built with tensorflow) It handles model versioning so you can esily handle new versionof your production

### In graph vs between graph replication

We can also place neural network on every device. like an ensemble. we then use the ensemble prediction

This can be handled in two major ways
1. we can create on big graph containing every neural network, each pinned to a differnet device plus the computations needed to aggregate the individual preconditions from all the neural networks then we creat one session from any server for taking care of individual predictions thisis called in-graph prediction

2.alternately we can create multiple graphs for each neural network and handle synchronization between graphs ourselves this is called between graph replication

These solutionsa have their pros and cons
in graphreplication is somewaht simpler since you dont have to manage multiple clients and multiple queues how even in between is a bit better ti organise
we can also add a timeout through passing a RunOptions with timeout_in_ms

### Model parallelism

running single neural network to mulptiple devices. 
IT is highly dependent on the architecture chosen like CNN or RNN

### Data parallelism

Another way is to replicate it on each device running a training step by using different mini batch

The update of parameter to central model can happen in two ways
1. Synchronously :  aggregator waits for all gradients to be availabel before computing the average and applying the result (using theaggregated gradients to pdate the model parameters ) once a replica hasfinished computin itmust wait for the parameters to be updated, to save time we can also ignore the gradients of slowest

2. Asynchronously : The weights are immediately updated. Although it iissurprising it works. It has a stake gradient problems that is when it these gradients are seriously unable to compute

There ar ea few ways to reduce the effects of stale gradients:
Reduce learning rate
Drop stale gradients or scale them down
Adjust mini batch size
Start the first few epochs using just one replica (Called warmup phase)

However it also creates a problemof bandwidth saturation becuse of the time spent in moving data in and out of gpuram

saturatoin ismore sever for large dense models It is less sever for small models since the gradients are typically moslet zeros. 

We can avoid this problem by Grouping out GPU on a few servers rather than scattering them accross many servers
Shard the parameter

Drop the model parameters float precision from 32 bits to 16 bits. This will cut in half the amount of data to transfer without much impact on convergence rate or model performance

Although 16 bit precision is minimum we can also drop the precision with a process known as quantizing the network which helps to run mobile networks on mobile phones.

### Steps for tensorflow implementation

1. We need to choose wether we waant in graph replication or between graph , synchronous or async updates

2. With in-graph replication we build one big graph containing all the model replicas placed on different device and a few nnodes to aggregate all the gradients and feed them to an optimizer. Your code opens a session to a cluster and simply runs the training operaton repeatedly

With in graph with async we create one graph, with one optimizer one therad repeatedlyrunning the the replicas optimizer

with in between graph and sync we run a mi=ultiple independent clients, each trining trh model replica as if it were alone in the world

with between graph and syn we run multiple clients , each training model replica based on shared parameters but this time we wrap the optimizer with a syncReplicaOptimizer The chief. It aggregateds and applies gradients. This approach also support spare replicas

# Exercises

1. 








 








