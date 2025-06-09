// === svm_model.rs ===
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct SVMRegressor {
    pub centers: Vec<Vec<f64>>,     // Support vectors (centres pour noyau RBF)
    pub alphas: DVector<f64>,       // Coefficients
    pub gamma: f64,                 // Paramètre du noyau RBF
    pub b: f64,                     // Biais 
    pub epsilon: f64,
    pub c: f64,

}

impl SVMRegressor {
    pub fn new(n_centers: usize, gamma: f64, epsilon: f64, c: f64) -> Self {
        SVMRegressor {
            centers: Vec::new(),
            alphas: DVector::zeros(n_centers),
            gamma,
            epsilon,
            c,
            b: 0.0,
        }
    }
    fn rbf(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-self.gamma * dist_sq).exp()
    }

    fn compute_kernel_matrix(&self, x: &Vec<Vec<f64>>) -> DMatrix<f64> {
        let n = x.len();
        let mut kernel = DMatrix::zeros(n, self.centers.len());
        for (i, xi) in x.iter().enumerate() {
            for (j, cj) in self.centers.iter().enumerate() {
                kernel[(i, j)] = self.rbf(xi, cj);
            }
        }
        kernel
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        // Choix aléatoire des centres
        let mut rng = thread_rng();
        self.centers = x.choose_multiple(&mut rng, self.alphas.len()).cloned().collect();

        let phi = self.compute_kernel_matrix(x);
        let phi_t = phi.transpose();
        let phi_t_phi = &phi_t * &phi;

        if let Some(inv) = phi_t_phi.try_inverse() {
            let y_vec = DVector::from_vec(y.clone());
            self.alphas = inv * (&phi_t * y_vec);
        } else {
            eprintln!("Erreur : matrice non inversible pour la résolution analytique");
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let kernel_values: Vec<f64> = self.centers.iter().map(|c| self.rbf(x, c)).collect();
        let kernel_vector = DVector::from_vec(kernel_values);
        self.alphas.dot(&kernel_vector) + self.b
    }
}
