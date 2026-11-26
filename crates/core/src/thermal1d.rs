//! 1D transient heat conduction, constant properties.

/// Thermal properties of a homogeneous material (SI units).
pub struct Material {
    /// Thermal conductivity [W/(m·K)]
    pub k: f64,
    /// Density [kg/m³]
    pub rho: f64,
    /// Specific heat [J/(kg·K)]
    pub cp: f64,
}

impl Material {
    /// Thermal diffusivity α = k / (ρ·cp) [m²/s]
    pub fn diffusivity(&self) -> f64 {
        self.k / (self.rho * self.cp)
    }
}

/// Plane slab discretized with a uniform grid of nodes
pub struct Slab {
    pub material: Material,
    /// Thickness [m]
    pub length: f64,
    /// Number of nodes, both faces included (at least 2)
    pub nodes: usize,
}

impl Slab {
    /// Node spacing [m]
    pub fn dx(&self) -> f64 {
        assert!(
            self.nodes >= 2,
            "Slab needs at least 2 nodes, got {}",
            self.nodes
        );
        self.length / (self.nodes - 1) as f64
    }

    /// Position of each node
    pub fn positions(&self) -> Vec<f64> {
        (0..self.nodes).map(|i| i as f64 * self.dx()).collect()
    }
}

/// Solves a tridiagonal system in place (Thomas algorithm).
///
/// `a`: sub-diagonal, `b`: diagonal, `c`: super-diagonal (all of length n;
/// `a[0]` and `c[n-1]` are ignored). `d`: right-hand side, overwritten with the solution.
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64]) {
    let n = d.len();
    let mut c_prime = vec![0.0; n];
    c_prime[0] = c[0] / b[0];
    d[0] /= b[0];
    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d[i] = (d[i] - a[i] * d[i - 1]) / m;
    }
    for i in (0..n - 1).rev() {
        d[i] -= c_prime[i] * d[i + 1];
    }
}

/// Transient conduction in `slab` with fixed temperatures on both faces
/// (Dirichlet), backward Euler in time. Returns nodal temperatures [°C]
/// after `steps` time steps of size `dt` [s].
pub fn solve(
    slab: &Slab,
    t_initial: f64,
    t_left: f64,
    t_right: f64,
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    let n = slab.nodes;
    let dx = slab.dx();
    let r = slab.material.diffusivity() * dt / (dx * dx);

    let mut t = vec![t_initial; n];
    t[0] = t_left;
    t[n - 1] = t_right;

    // Interior rows: -r·T[i-1] + (1+2r)·T[i] - r·T[i+1] = T_old[i]
    let mut a = vec![-r; n];
    let mut b = vec![1.0 + 2.0 * r; n];
    let mut c = vec![-r; n];
    // Boundary rows: T = fixed value (already stored in t)
    for i in [0, n - 1] {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 0.0;
    }

    for _ in 0..steps {
        solve_tridiagonal(&a, &b, &c, &mut t);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONCRETE: Material = Material {
        k: 1.6,
        rho: 2300.0,
        cp: 900.0,
    };

    #[test]
    fn diffusivity_of_concrete() {
        let alpha = CONCRETE.diffusivity();
        assert!((alpha - 7.729e-7).abs() < 1e-9);
    }

    #[test]
    fn slab_grid() {
        let slab = Slab {
            material: CONCRETE,
            length: 0.3,
            nodes: 4,
        };
        assert!((slab.dx() - 0.1).abs() < 1e-12);

        let x = slab.positions();
        assert_eq!(x.len(), 4);
        assert_eq!(x[0], 0.0);
        assert!((x[3] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn tridiagonal_3x3() {
        // [2 1 0; 1 2 1; 0 1 2] · x = [4 8 8]  →  x = [1 2 3]
        let a = [0.0, 1.0, 1.0];
        let b = [2.0, 2.0, 2.0];
        let c = [1.0, 1.0, 0.0];
        let mut d = [4.0, 8.0, 8.0];
        solve_tridiagonal(&a, &b, &c, &mut d);
        for (got, want) in d.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn steady_state_is_linear() {
        // After a long time the profile between 100 °C and 0 °C is a straight line.
        let slab = Slab {
            material: CONCRETE,
            length: 0.1,
            nodes: 11,
        };
        let t = solve(&slab, 0.0, 100.0, 0.0, 3600.0, 200);
        for (ti, x) in t.iter().zip(slab.positions()) {
            let expected = 100.0 * (1.0 - x / 0.1);
            assert!(
                (ti - expected).abs() < 1e-3,
                "x={x} T={ti} expected={expected}"
            );
        }
    }
}
