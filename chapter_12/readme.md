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