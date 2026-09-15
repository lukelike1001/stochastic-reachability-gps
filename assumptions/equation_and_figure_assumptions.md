# Equation/Figure Assumptions
A list of assumptions made for implementing certain equations and figures from the paper.

## Section 2.2
### Equation 1
* None, definition of zonotope

### Equation 2
* None, definition of p-zonotope

### Evaluating a p-zonotope at a point
* Equation 26 asks for $\mathcal{Q}_k(\mathbf{q}_k)$, but the paper never says what evaluating a p-zonotope at a point means numerically.
* I'll take it as the highest density any distribution in the set can give that point, so $\sup_{\|\beta\|_\infty \le 1} \mathcal{N}(\mathbf{q}; \mathbf{c} + \mathbf{G}\beta, \Sigma)$.
* That is the same as the shortest Mahalanobis distance from the point to the zonotope $\mathcal{Z}(\mathbf{c}, \mathbf{G})$ under $\Sigma$, which is a small box-constrained least squares problem.

## Section 3.2
### Equation 3
* Equation 3 represents the true state, and none of the sensors or filters can ever record or calculate this.

### Equation 6
* The paper doesn't write out H, so we assume $H = [I_2 \; 0_2]$, since GPS gives position only.

### Equation 14
* This equation is used to derive Equation 17, so we don't need to implement this.

### Equation 47
* We assume that $Q = \Gamma_k @ W_{k-1} * \Gamma_k^T$
* This means $\Gamma_k = I_4$.
* State order is $(p_x, p_y, v_x, v_y)$, which is what makes Equation 47's block structure line up.

## Section 4.1.2
### Figure 4
* The true position curve start and end position aren't provided, nor are the start and end times. Thus, I'll estimate the points.
* I believe that the true position curve starts at (0, 0) at t=0, reaches its rightmost extreme (30, 30) at t=3, before curving westward to end at (25, 65) at t=6.
* There is no adversarial attack for Figure 4, so the Chimera-authenticated GPS estimate, the unauthenticated GPS estimate, and the self-contained position estimate should all roughly follow the curve, with some added noise.

### Figure 5
* Same problem as Figure 4 where no exact values are provided.
* Like Figure 4, I believe that the true position curve starts at (0, 0) at t=0, reaches its rightmost extreme (30, 30) at t=3, before curving westward to end at (25, 65) at t=6.
* The adversarial unauthenticated GPS curve, in my opinion, start at (0, 0) at t=0, (45, 30) at t=3, (65, 80) at t=6.
* Chimera-authenticated GPS and self-contained sensor should both follow the true curve, while the unauthenticated GPS follows the adversarial curve.

## Section 4.2.2

### Figure 6
* I'll provide estimates after finishing the linear SR-KF case.

### Figure 7
* Ditto as Figure 6.