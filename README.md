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
