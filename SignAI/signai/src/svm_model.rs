use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Serialize, Deserialize};
use serde_json;

#[derive(Serialize, Deserialize)]
pub struct SVMClassifierRBF {
    pub support_vectors: Vec<Vec<f64>>,
    pub alphas: DVector<f64>,
    pub labels: Vec<f64>,
    pub gamma: f64,
    pub b: f64,
    pub c: f64,
    pub epochs: usize,
    pub lr: f64,
}
impl SVMClassifierRBF {
    pub fn new(n_support: usize, gamma: f64, c: f64, lr: f64, epochs: usize) -> Self {
        Self {
            support_vectors: Vec::new(),
            alphas: DVector::zeros(n_support),
            labels: Vec::new(),
            gamma,
            b: 0.0,
            c,
            epochs,
            lr,
        }
    }

    fn rbf(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-self.gamma * dist_sq).exp()
    }

    fn compute_kernel(&self, xi: &Vec<f64>) -> DVector<f64> {
        let values: Vec<f64> = self.support_vectors.iter().map(|sv| self.rbf(xi, sv)).collect();
        DVector::from_vec(values)
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        // Choisir des vecteurs support aléatoirement
        let mut rng = thread_rng();
        let chosen: Vec<_> = x.choose_multiple(&mut rng, self.alphas.len()).cloned().collect();
        self.support_vectors = chosen;
        self.labels = y.clone();

        // Descente de gradient sur les alphas (formulation duale simplifiée)
        for _ in 0..self.epochs {
            for i in 0..x.len() {
                let xi = &x[i];
                let yi = y[i];
                let k_vec = self.compute_kernel(xi);
                let margin = self.alphas.dot(&k_vec) + self.b;
                let error = yi * margin;

                if error < 1.0 {
                    for j in 0..self.alphas.len() {
                        self.alphas[j] += self.lr * (yi * k_vec[j] - self.c * self.alphas[j]);
                        self.alphas[j] = self.alphas[j].clamp(0.0, self.c);
                    }
                    self.b += self.lr * yi;
                }
            }
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let k_vec = self.compute_kernel(x);
        let value = self.alphas.dot(&k_vec) + self.b;
        if value >= 0.0 { 1.0 } else { -1.0 }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(file, &self)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let model = serde_json::from_reader(file)?;
        Ok(model)
    }
}
 pub fn evaluate_accuracy_svm(
        model: &SVMClassifierRBF,
        X_test: &Vec<Vec<f64>>,
        y_test: &Vec<f64>,
    ) -> f64 {
        let mut correct = 0;
        for (x, &y_true) in X_test.iter().zip(y_test.iter()) {
            let y_pred = model.predict(x); // float
            let y_pred_label = if y_pred >= 0.5 { 1.0 } else { 0.0 };
            if (y_pred_label - y_true).abs() < 1e-6 {
                correct += 1;
            }
        }
        correct as f64 / y_test.len() as f64
    }