# Permeability diffusion problem

$$
- \nabla \cdot (\kappa \nabla u) = 0
$$
with boundary conditions
$$
u(x = 0, y) = f(y, e_1) = \exp\left(-\frac{1}{2\sigma_w} (y - e_1)^2\right)
$$
$$
u(x = 1, y) = g(y, _2) = - \exp\left(-\frac{1}{2\sigma_w} (y - e_2)^2\right)
$$
using the finite element (FE) method with second-order Lagrange elements on a mesh of size $ h = \frac{1}{64} $ in each coordinate direction, where $ \kappa $ is a custom 2D matrix.

$$
c(x, z) = \sigma_v^2 \exp \left[ \frac{-\|x - z\|^2}{2 \ell^2} \right] \quad \text{for } x, z \in \Omega,
$$


$$
m(x, \mathbf{m}) \approx \sum_{i=1}^{n_m} m_i \sqrt{\lambda_i} \phi_i(x),
$$