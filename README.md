# EXERCISE-SEPERABLE-CONVOLUTIONS-PERFORMANCE

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree. (Solution of the exercise and adaptation as a repository: Andrés R. Bucheli.)

Seperable convolutions performance.

## Exercise: Seperable Convolutions Performance
For this exercise your first task will be to calculate the total number of FLOPs in a model that uses separable convolutional layers. The architecture of this <code>sep_conv</code> model is
give below. Your second task will be to then create an inference engine pipeline that can run the model.

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-SEPERABLE-CONVOLUTIONS-PERFORMANCE/master/l3-slides-sep.jpg)

## Task 1: Calculate Model FLOPS
<strong>Layer 1: Separable Conv2D</strong>
Input shape: 1x1x28x28
Kernel shape: 3x3
Number of kernels: 10

Depthwise Layer:
The shape for a single dimension will be = (28-3)+1 = 26
So our output shape will be 26x26
Because we have 1 input channel, our actual output shape will be 1x26x26
FLOPs: 1x26x26x3x3x1x2 = 12,168
Pointwise Layer:
Input Shape = 1x26x26
No. of kernels = 10
Output Shape = 10x26x26
FLOPs: 10x1x1x1x26x26x2 = 13,520
Total FLOPs: 12168+13520 = 25,688

<strong>Layer 2: Separable Conv2D</strong>
Input shape: 10x26x26
Kernel shape: 3x3
Number of kernels: 5

Depthwise Layer:
The shape for a single dimension will be = (26-3)+1 = 24
So our output shape will be 24x24
Because we have 10 input channel, our actual output shape will be 10x24x24
FLOPs: 10x24x24x3x3x1x2 = 103,680
Pointwise Layer:
Input Shape = 10x24x24
No. of kernels = 5
Output Shape = 5x24x24
FLOPs: 5x1x1x10x24x24x2 = 57,600
Total FLOPs = 103680 + 57600 = 161,280

<strong>Layer 3: Fully Connected</strong>
Number of neurons: 128

Input shape: 24x24x5: 2880
FLOPs: 2880x128x2 = 737,280

<strong>Layer 4: Fully Connected</strong>
Input Shape: 128
Output Shape: 10

FLOPS: 128x10x2 = 2560

## Task 2: Completing the Inference Pipeline
Complete the <code>inference.py</code> python script on the right.

Remember to source the OpenVINO environment before running the python script.

To run the code><inference.py</code> file, you can use the command:
<code>python3 inference.py</code>

<strong>Note:</strong> You may get a warning about OpenVINO using a different Python version. You can ignore this warning, the inference should still run fine.

<pre><code>
from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Getting model bin and xml file
model_path='sep_cnn/sep_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

model=IENetwork(model_structure, model_weights)

core = IECore()
net = core.load_network(network=model, device_name='CPU', num_requests=1)

input_name=next(iter(model.inputs))

# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)


input_dict={input_name:input_img}

start=time.time()
for _ in range(10):
    net.infer(input_dict)


# TODO: Finish the print statement
print("Time taken to run 10 iterations is: {} seconds".format(time.time()-start))
</code></pre>





## Solution of the exercise and adaptation as a Repository: Andrés R. Bucheli.


