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
}
