---
layout: post
title: Computational Model of Memory
subtitle: What is memory and how to make computational model for it?
tags: [computational neuroscience]
---

## How do humans store memory?
I am always fascinated about the human brain. There are so much that we don't know! One interesting thing I want to investigate is how we store memories and facts.
There is a famous incident in neuroscience history: people in early 1900's found out there are neurons in the brain that fires that correspond to certain shapes
in the retina. For example, there are neurons for points, for lines. This is of course the inspiration for Convolutional neural net, something we will not be talking about here.
However, there was only a question about higher level shapes. If every neuron is responsible for one shape or image, we will need exponential amount of neurons for more and 
more complex shapes. Neuroscientists had a quest for finding the so-called "grandmother neuron", the neuron that fires when you see your grandma. They never found it. 

As more researches were done, neuroscientists found that a fact or your grandma's face is not stored in a single neuron, but stored in a network of neurons and their connections. For example, when some parts of a person's brain are damaged, he/she seems to be able to remember facts as he/she did before. However, significant brain damage will make a person take more cues to remember certain events or facts. The theory of memory is this: say you are trying to remember what happened in your 18-th birthday, you don't go to one neuron that "remembers everything about that event". You go to a neuron that remembers something about the event: say you remember "I am with my family", then that triggers other neurons to fire - you remember that you were at a restaurant. Maybe you were eating sushi. You then remember the name of the restaurant, what food you ordered. Then you started to remember the whole event and every detail of it and what gifts you received. The things is, you store details distributively across the whole network of neurons.  

## How does computer store memory?
Well, for folks who are not familiar with computers, computers store information in a straightforward way. Imagine a huge cabinet of boxes, each one has a number written on it. And every box has some data in it. All computers can do is to go fetch a piece of data in a box with a specific number on it. 

## Content addressable memory
What is the difference between computer memory and human memory? Well humans can remember events with partial corrupted information, while computers can't. The form of human memory is what neuroscientists call the content addressable memory. And I've been thinking about how to implement content addressable memory in computers. And it turns out, this has been done by a physicist/neuroscientist - Hopfield.

## Spin glass and Hopfield's Genius
Hopfield was in his heart a physicist. In 1980's he was inspired by a physical system called spin glass, where each element in the system is a little magnet. And every magnet will change its direction based on all the forces it is feeling based on every other magnets. If you have a spin glass and let loose the system, every magnet will change its direction and will finally come to a equilibrium state where no magnet will flip. In this equilibrium state, when one magnet is pointing its direction against the rest of the network, it will flip back. This is sort of like memory, Hopfield thought. When some corrupted signal comes through and some magnets went "rogue", they will be forced by the rest of the network to go back to their "correct" state, the state that minimizes the energy of the whole system. In 1982, Hopfield wrote a paper about this physical system and its implication for systems that can simulate content addressable memories. This network that can remember things that he invented is now called Hopfield network.

![](/images/hopfield-network-example.jpg "An example of how Hopfield Network can remember pattern of digit 3 and successfully recall it given a partial 3 pattern")

## The setup and the challenge
So the setup of the problem is simple: given some bit patterns that represent the memories, can our model remember those patterns, and given some corrupted version of the pattern, recall the correct pattern? For example, if we are given a 25-vector that in 5 x 5 shapes like a letter T, like below:  
[1, 1, 1, 1, 1,  
 -1,-1,1,-1,-1,   
 -1,-1,1,-1,-1,   
 -1,-1,1,-1,-1,  
 -1,-1,1,-1,-1]  
  And another vector that is shaped like C, like:  
  [1, 1, 1, 1, 1,  
   1,-1,-1,-1,-1,   
   1,-1,-1,-1,-1,   
   1,-1,-1,-1,-1,  
   1, 1, 1, 1, 1],  
   Can we give the correct pattern when given a slightly different T or C shape? 

   {% include info.html text="We choose 1 to represent on neurons and -1 to represent off neurons because if we choose 0 to represent off neurons then Hebbian learning rule (see following for description of Hebbian learning rule) will only increase strengths of connections and never decrease their strengths." %}
## Hebbian learning rule
> Hebb: Neurons that fire together wire together.

Okay, so given two patterns, how does our model remember them? We use the famous Hebbian learning rule. We first design a weight matrix that represent the connection strengths between every neuron and every other neuron. Just like in spin glass, every neuron's firing influences every other neuron's firing, and the strength of the synapse is the importance of that connection. in the remembering phase we want to find the weight matrix such that when we let the network loose it will find the correct pattern. Hebbian learning is just saying the neurons that fire together, wire together. For example, if neuron A and neuron B are both on (their values are both 1), then their connection is strengthened. If neuron A is 1 and neuron B is -1, then their connection is weakened. We do this for every neuron pair in both T shape and C shape vectors. The update rule is given by the following matrix equation:  
$$
W(t+1)= W(t) +  \eta(ss^T - I)
$$  
Where $\eta$ is the learning rate, $s$ is one pattern vector we want to remember. We simply initialize a zero matrix, and do the above update rule for every pattern vector we want to remember. And voila, we will have the weight matrix we want.
we minus the identity matrix since no neuron connects to itself. the outer product expresses exactly the pairwise products between every pair of neuron that we need. 

## The inference rule
So given the weight matrix we learned, and a corrupted pattern, how do we recall? Simply as following:
every neuron i turns on if $\sum_{j\ne i}^{n} W_{ij} s_{j} > 0$, else if turns off.  where $s_{j}$ is the j-th neuron (1 if on, -1 if off), $W_{ij}$ is the connection strength from j-th neuron to the i-th neuron. In matrix form we have:  
$$
s = sign(Ws)
$$,  
where sign is just a elementwise operation on every entry in the vector, and if element > 0 it will return 1, else return -1.  
In python we have:
```python
for i in range(100):
   # need to import numpy to use @ as matrix multiplication operator
   # this is one iteration where we turn every neuron on or off based
   # on its total input from all other neurons.
  inputs_vector = W @ corrupted_C # calculate input to each neuron by matrix multiplication
  corrupted_C = [1 if x > 0 else -1 for x in inputs_vector] # mimic threshold function
  # after many iterations corrupted_C vector will converge to the correct C pattern.
```
It turns out if we have a corrupted partial C-shape or T-shape vector now, and we apply the above rule until convergence, we will get
the complete memory pattern!!! Isn't this amazing!!

## the limitation of Hopfield Network
As Hopfield analyzed, the capacity of the network is pretty low: it is 0.15N. In other words, a network with N neurons can remember at most 0.15N memory patterns reliably. There is of course a way to expend this capacity. In fact the whole field research of Boltzmann machine was done originally to increase the memory capacity of Hopfield network. That is another story for another time. I hope you enjoy this. 

[the above illustration in python code](https://github.com/wenjunsun/personal-machine-learning-projects/blob/master/boltzmann-machine/Hopfield_network.ipynb)

[Play with Hopfield Network](http://faculty.etsu.edu/knisleyj/neural/neuralnet3.htm)
