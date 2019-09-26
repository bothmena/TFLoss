## Task

#### Definition

Suppose we want to predict the value of a 3-dimensional vector y and we have a predictive model that, for each input, 
returns not just one, but K different estimates of this vector. Write a loss function in tensorflow that takes a batch 
of these predictions and a batch of target values of y (with batch size M), and outputs a batch of losses. Please use 
numpy-style docstrings!

#### My question

for each input **x**, we have a target vector **y** a vector of shape (3,).
We have a predictive model, and for each input **x**, it outputs **y_hat** a vector of shape (**K**, 3).

My task is to write a loss function that takes two arguments:
- **y_batch**: a vector of shape (**M**, 3) and I assume that for the same input, this vector contains **M** 
"3-dimensional vectors" which are the target of the input repeated **M** times, so if **M** = 2 then 
**y_batch** = [**y**, **y**] 
- **y_hat_batch**: a vector of shape (**M**, 3).

The loss function should return a vector of shape (**M**, 1).

---

From the description, the task looks like a regression task, but it can also be a 3-class classification.

Can I implement any existing and already implemented loss like MAE, MSE for regression tasks or Cross-Entropy for 
classification?

I assume that I should not use numpy to make all the calculation and implement everything from scratch.

#### Answer of my question
Assume that the inputs to your loss function is a batch of targets, of shape (**M**, 3), and a batch of stacks of 
estimates, of shape (**M**, **K**, 3). 

Come up with a reasonable way of scoring such a function.

---
---

# Documentation and Discussion

This task was developed using the TDD approach, but the written tests do not cover all cases.

The repository have 2 branches:
- master: contains an implementation of the loss function using numpy arrays as a data structure.
- tf-api: contains an implementation of the loss function using TensorFlow Tensors as a data structure.

So as mentioned above in the task definition, the loss function has two inputs: y_batch of shape (M, 3) and y_hat_batch of shape (M, K, 3).

The current implementation of the loss is that we compute the mean square error between the input target and all the K predictions. 

A reduction method can also be specified, and it can take 4 different values:
- None: default behavior, the loss function returns an array of shape (K, 1) which is the loss function between all predictions and the real value (target).
- min: the loss function only returns a scalar which is the minimum value of the loss for all K predictions
- max:  the loss function only returns a scalar which is the maximum value of the loss for all K predictions
- mean: the loss function only returns a scalar which is the mean value of the loss for all K predictions

In my opinion, using the mean reduction is the best option, because we will be able to minimize all the predictions
the model is making and because mean is easier to differentiate than min and max. Also the mean gives us a better
understanding of the performance of the model, especially when compared with previous loss values.
