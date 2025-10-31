# SignalProcessingDemos
A couple of demo's visually showing the effects of amplification, filters on signal processing chains or how a discretized PID controller behaves as a result of signal delay/

PID/main.py shows P, I and D action, setpoint changes, and the influence of time delay on a noisy sinusoidal input.

Signal Chain /main.py shows the effects of gain, low-pass cutoff frequency, capture frequency and discretization on a perfect sinusoidal input. This sinusoid is the input for the system, but some noise is added to mimic a real system. 
