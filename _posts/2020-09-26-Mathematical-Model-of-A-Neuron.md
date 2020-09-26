---
layout: post
title: Mathematical Model of A Neuron
subtitle: using differential equation to describe voltage dynamics of a neuron
readtime: True
tags: [computational neuroscience]
---
First off, what do we mean by a mathematical model of a neuron? What are we describing? What characteristics of a neuron
do we want to capture?

To be fair, what a real neuron does is very complex!!! There are rich dynamics involved -- there are millions of ion channels on a single neuron. Different channels transport different ions (which includes potassium, sodium and much more) at different rate. Different ion channels also have different properties. Some ion channels are electricity gated, which means the membrane potential has an effect on how likely it is going to open. (Membrane potential is the voltage within the cell.) When a neuron is at resting state it has a voltage potential of about -65 mV. 

So, what can we abstract from this infinitely complex little neuron, and describe what it is doing? If we can only know one
number that describes the neuron at a specific time, what do we want to know? Well, it will probably be the membrane potential
(voltage) of the neuron. Since we all know that when a neuron's voltage reaches a certain threshold, it fires its signals to its friends,
which is the most basic computation done by neurons.

So neurons compute by firing. And to fire its voltage inside the cell has to reach a certain threshold. So we want to model the voltage. Okay. But how?

So we can think of cell membrane of a neuron as an insulator (something that doesn't allow electricity to flow), and there are charges
on both sides of the membrane - net positive on the outside and net negative on the inside. The only way charges go in and out of the
cell is via ion channels. And the more charges are on both sides of the neuron, when you open the gate, the more they are going to flow,
right? This is described by $$Q = CV$$, where $$Q$$ is the number of charges on the neuron membrane, $$C$$ is the capacitance (how good is this neuron at storing charges - the more surface area of the neuron, the more charges it can store of course), and $$V$$ is the voltage across the membrane.

Now let's differentiate this formula on both sides with respect to time, then we get:
\\[ \frac{dQ}{dt} = C\frac{dV}{dt} \\]
 Since $$\frac{dQ}{dt} = I$$, which just means current = change in number of charges with respect to time, we get
\\[ I = C\frac{dV}{dt} \\]
So what do we have now? We derived a simple fact: the change in voltage of the cell membrane = the current that comes in to the cell. In other words,
to model voltage we need to model the incoming current to the cell. How? 
Well, every neuron has a lot of pumps to keep its resting potential. (neurons work all the time, even when they are not firing!). Since 
neuron's resting potential is negative, that means neuron is constantly pumping out positive ions (typically sodium ions) to keep its negativity.
So that is one way current can get out of the cell. the other thing we model is the synapse. Currents come in through the synapse via other
cell's signaling. factoring those factors and put those inside the above formula we have:
\\[  C\frac {dV}{dt} = -(Leakage) + synapse + external input current \\]
\\
