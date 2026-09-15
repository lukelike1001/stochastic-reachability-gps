# Filter Assumptions
A list of assumptions made for implementing filters, their probabilistic zonotopes (p-zonotopes) and errors from the paper.

## General
* Only the simulator gets to touch the true state. It needs truth to generate measurements and to score the panel (c) bounding ratio, but no filter or sensor may read it.
* We propagate the true state with the discrete A and B rather than an exact continuous integral. Otherwise the truth and the filters disagree by a small amount that looks like drift instead of noise.

# Self-Contained Stochastic Reachability Kalman-Filter (Self-Contained SR-KF)
NOTE: General outline that you don't need to use exactly as-is.

## Self-Contained p-Zonotope
* Section 4.1.1 gives the sensor bounds in metres and metres per second, not in acceleration units, so the paper already folded the sensor error through B into state space. We corrupt the state, not the acceleration reading.
* The paper says what the set $\mathcal{W}_k$ is but never how to draw a realization from it. We draw the Gaussian part from Q and the bias uniformly from the box.
* We redraw the bias every step, since Equation 20 tacks on a fresh generator every step. It could also be one bias held for the whole epoch. The zonotope widths in Figure 4(b) should tell us which.

## Self-Contained State Estimate
* The paper initializes from a Chimera-authenticated least squares fix, but doesn't give the numbers, so we pick a starting state ourselves.
* Figures 4 and 5 only cover one 6 second epoch, so Chimera fires at t=0 and never again. Re-anchoring mid-run is out of scope for now.

## Self-Contained Error
* The Chimera fix comes from a least squares solve, so it is authentic but not exact. The initial error p-zonotope is not the zero set.
* We set it to the Chimera fix's own error set. Chimera proves a signal is genuine, it doesn't make it more accurate, so that error is the same +/-0.5 m and 5 m as any other GPS fix.
* Equation 25 backs this up. At k=0 the statistic is the difference of two independent GPS-grade fixes, so its variance is 25 + 25 and its bias spans +/-1.0 m. Only a non-zero initial error set makes $\mathcal{Q}_0$ come out that wide. Starting from the zero set makes the overbound too narrow by a factor of root 2, which would blow the false alarm budget right where Figure 4(a) shows CAR still holding above 0.997.
* The self-contained error set only grows, since there is no measurement update to shrink it.

# Unauthenticated GPS p-Zonotope Estimator
* This is not a filter. Equation 21 uses the raw received measurement, so there is no state to propagate.
* Its error p-zonotope is constant in k, since Section 4.1.1 gives one bias bound and one variance with no time dependence.
* We draw the GPS bias once per run and hold it, unlike the process bias. A bias that changed every step would just be more noise, and nothing in the paper concatenates generators for it.
* The Chimera receiver and the unauthenticated receiver draw independent bias realizations, even though they share one error model and one physical receiver. The Chimera fix comes from the previous epoch, 6 seconds and roughly 60 m earlier with different satellite geometry, so we treat the multipath as redrawn. This one is load bearing: the initial error argument above needs $\mathbf{q}_0$ to have variance 25 + 25, and a shared bias would partly cancel and leave $\mathcal{Q}_0$ too wide.
* Spoofing changes the measurements and nothing else. The attacker never touches the declared error bounds or variance, since those come from calibration rather than from the signal. If the spoofed sensor widened its own declared error, the detector would quietly accommodate the attack and Figure 5 would show no detection at all.

# Fused SR-KF
* Equation 17 needs the Kalman gain, so we have to run the point-valued covariance recursion even though no figure ever plots it.
* The fused filter starts from the same Chimera fix and the same initial error p-zonotope as the self-contained filter. Section 4.1 says the fix is what "our SR-KF and SR-EKF leverage for initialization" without separating the two filters, and Figure 3 runs one Chimera arrow into both boxes, but using the same $\tilde{\mathcal{X}}_0$ for both is our reading.
* Whether the fused error set settles or keeps growing is still open. Section 4.2.2 only says the GPS set is constant and the self-contained set grows, and says nothing about this one. Do not use it as a correctness check until we have something to compare against.

# SR Spoofing Detector
* $P_{FA} = 0.003$ and CAR $= 0.997$ are the 1D 3-sigma numbers, but $\mathbf{q}_k$ is 2D.
* We assume the threshold is a 3-sigma Mahalanobis radius, since the paper keeps calling the plotted sets "3-sigma zonotopes."
* Equation 26 compares $\mathcal{Q}_k(\mathbf{q}_k)$ against $P_{FA}$, but a density carries units of one over metres squared and its size depends on the determinant of Sigma, so that comparison is not well posed. Maximising the density and minimising the Mahalanobis distance are the same thing, so we threshold the distance instead and get the same decision.

# Output SR Filter Switch
* The switch latches. Section 3.1 says the output relies on the self-contained filter "until it can re-authenticate the received GPS measurements", so one authentic looking step does not undo a spoof call. The captions of Figures 5(b) and 7(b) agree, describing the switched estimate as matching the self-contained one for the latter half or the majority of the trajectory.
* The detector itself does not latch. Equation 26 recomputes $d_k$ every step, and that per-step decision is what panels 5(a) and 7(a) plot.
* Within one 6 second epoch the latch never clears, since re-authentication only arrives at the next epoch.
* The fused filter keeps stepping while it is latched out, so it is ready if re-authentication arrives. The paper doesn't say either way.
