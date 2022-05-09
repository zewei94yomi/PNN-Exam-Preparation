# PNN final exam preparation


## Intro

The final exam of module **Pattern Recognition, Neural Networks and Deep Learning** 
requires lots of calculation by hand. Though the module lecturer said Matlab or Python
programs will be unnecessary for the exam, it's convenient to let the computer do the 
calculation.

Python programs in this repository is mainly based on tutorial questions. In each
directory, there is a tutorial question sheet and an answer script. The answer script
only contains solutions to calculation questions that are likely to appear in the 
exam. 

## How to run
Each `.py` file has a `main` method. Just simply run it. 

You shall change parameters in the method definition.

## Week1: Introduction to Pattern Recognition

- Q8: Error rate, Accuracy, Recall, Precision, F1-Score

## Week2: Discriminant Functions

- Q1: Dichotomizer
- Q2, Q5: Dichotomizer (augmented vectors)
- Q4: 2D Quadratic Discrimination
- Q6, Q7, Q9, Q10: Perceptron Learning
    - Input data: Augmented Notion + Sample Normalization
    - Choose 'Sequential' or 'Batch' learning
- Q11: Multiclass Sequential Perceptron Learning
- Q14: Widrow-Hoff Learning
    - Choose 'Iteration' or 'Epoch' weights updating
- Q15: K-Nearest Neighbor (from `sklearn`)

## Week3: Introduction to Neural Networks

- Q2: Neuron Learning
    - Linear weighted sum + Heaviside activation function
- Q3, Q4, Q5, Q6: Delta Learning
    - Choose 'Sequential' or 'Batch' learning
  
## Week4: Multilayer Perceptrons and Backpropagation 

- Q5: Sigmoid

## Week5: Deep Discriminative Neural Network

- Q4: ReLU, LReLU, tanh, heaviside
- Q5: Batch Normalization
- Q9: Calculate output dimensions