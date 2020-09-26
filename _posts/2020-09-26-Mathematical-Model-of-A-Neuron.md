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

So now we derived a simple fact: the voltage across the cell membrane changes when there is electricity current that comes in to the cell. In other words, to model voltage we need to model the incoming current to the cell, which is composed of currents of 
different ions. 

Okay, now let's try to figure out what kinds of current are coming in/out of the cell. To keep things simple, we'll just model two
sources of the current -- the ionic pump and the synapse. 
- what does an ionic pump do? Well, it pumps ions into the cell to keep its resting potential. This just means that if the neuron's
resting potential is -65 mV, and its current potential is -30 mV, then there are pumps on the neuron that try to restore the voltage
back to -65 mV, by transporting more positive charges into the cell. We sometimes call this ionic pump the "leakage gate". So the
current this leakage gate contribute is simply $$-\frac{(V-V_{equilbrium})}{R}$$, by Ohm's law that current = voltage / resistance.
Minus sign was there because higher the voltage, the more positive ions are coming in, not going out of the cell, which decrease voltage,
not increase voltage. 
- what is a synapse? Synapse is the way neurons connect to each other. When the a presynaptic neuron fires, it releases chemicals
that bind to the surface of the postsynaptic neuron, which in term will increase the current coming in/out of the cell based
on the nature of the synapse. We model the current contributed by the synapse by its conductance (denoted by $$g_s(t)$$), which is simply how well this synapse conduct electricity. We'll think about this conductance is high when the presynaptic neuron is excited, and low when it is not. The current of a synapse is simply $$g_{s}(t)(V - V_{s})r_{m}$$, where $$V_{s}$$ is the equilibrium potential of this
synapse. -- for example, if $$V_{s}$$ is very negative, then this synapse is trying to make postsynaptic neuron's voltage more negative,
and if it is positive, then it is trying to make postsynaptic neuron's voltage more positive. $$r_{m}$$ is the membrane resistance at a unit area. 

Here is a nice picture of a synapse. 
[image of a synapse](https://cdn.britannica.com/37/54737-050-013849FC/nerve-impulse-transmission-synapse-arrival-neurotransmitter-release.jpg)

Okay, so now we have expressions of all the contributing forces to the change in voltage, we have:
\\[  C\frac {dV}{dt} = -\frac{(V-V_{equilbrium})}{R} + g_{s}(t)(V - V_{s})r_{m} \\]

Now we have a differential equation describing the voltage of a neuron! This is also called a "RC Circuit Model of neuron".
Hopefully that was not that confusing!