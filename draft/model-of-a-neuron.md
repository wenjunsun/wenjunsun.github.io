## mathematical model of a neuron
First off, what do we mean by a mathematical mode of a neuron? Well to answer that, we first need to think a little bit about what a neuron does.
Of course what a real neuron does is very complex!!! There are rich dynamics involved, with millions of ion channels on a single neuron. Different
ion channels transport different ions (which includes potassium, sodium and much more) at different rate. Different ion channels also have different
properties. Some ion channels are electricity gated, which means the membrane potential has an effect on how likely it is going to open. Membrane potential
is the voltage within the cell. When a neuron is at resting state it has a voltage potential of about -65 mV. So comparing to outside of the cell a neuron
is relatively negatively charged, and when its potential cross some threshold (say -30 mV), it "fires" its signal to its neighbors.
I can go on and on about this, but you get the idea. Simply put, a neuron receives signals from other neurons. When it receives
enough signals, it fires its own signal to other neurons.
So how do we model this? Hopefully you've noticed the key characteristic of a neuron we want to model -- the membrane potential. In
other words, if we model the neuron's within the cell voltage at each single moment, then we know what this neuron will do. And once
we know that, we can use the "integrate and fire" model to say that once its voltage passes this threshold its voltage increases
to the "spiking voltage", and then decrease to resting potential. 
Okay, so we want to model the voltage of a neuron, but how? Well, we use a simply differential equation. For people who don't like math, please
stick with me here. I promise it is not that bad. So there is a formula from your high school physics class that you probably don't remember -
$$Q = CV$$. To understand this formula, first imagine a insulated slice, which means electricity can't pass through it. So say you have a piece of
glass. Then you put some positive charge on one side of the glass, and some negative charge on the other side, then you know positive charge
attracts negative charge. And since they can't travel through glass, they are effectively "stuck" to the surface of the glass. And the formula
simply says the number of charges on the glass = the capacitence of the glass  * voltage that is being generated. Simple right? 
Now let's differentiate this formula on both sides with respect to time, then we get:
\\[ \frac{dQ}{dt} = C\frac{dV}{dt} \\]
hopefully you remember that $$\frac{dQ}{dt} = I$$, which just means current = change in number of charges with respect to time
\\[ I = C\frac{dV}{dt} \\]
So what do we have now? We derived a simple fact: the change in voltage of the cell membrane = the current that comes in to the cell. In other words,
to model voltage we need to model the incoming current to the cell. How? 
Well, every neuron has a lot of pumps to keep its resting potential. (neurons work all the time, even when they are not firing!). Since 
neuron's resting potential is negative, that means neuron is constantly pumping out positive ions (typically sodium ions) to keep its negativity.
So that is one way current can get out of the cell. the other thing we model is the synpase. Currents come in through the synapse via other
cell's signaling. factoring those factors and put those inside the above formula we have:
\\[  C\frac {dV}{dt} = -(Leakage) + synapse + external input current \\]
\\
