// On implémentes un SVM binaire avec noyau RBF.
//Il distingue deux classes uniquement, représentées par les labels +1 et -1.
//Il n'y a pas de gestion multiclasse OvA ou OvO pour l'instant.
//c'est un svm avec noyeau RBF gaussien

use nalgebra::{DVector};
use std::f64::EPSILON;

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
    pub fn new(gamma: f64, c: f64, lr: f64, epochs: usize) -> Self {
        Self {
            support_vectors: Vec::new(),
            alphas: DVector::zeros(0),
            labels: Vec::new(),
            gamma,
            b: 0.0,
            c,
            epochs,
            lr,
        }
    }

    fn rbf(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter()
            .zip(c.iter())
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum();
        (-self.gamma * dist_sq).exp()
    }

    fn compute_kernel(&self, x: &Vec<f64>) -> Vec<f64> {
        self.support_vectors
            .iter()
            .map(|sv| self.rbf(x, sv))
            .collect()
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        self.support_vectors = x.clone();
        self.labels = y.clone();
        self.alphas = DVector::zeros(x.len());

        for _ in 0..self.epochs {
            for i in 0..x.len() {
                let xi = &x[i];
                let yi = y[i];

                let k_vec = self.compute_kernel(xi);
                let sum: f64 = self.alphas.iter()
                    .zip(self.labels.iter())
                    .zip(k_vec.iter())
                    .map(|((&alpha_j, &yj), &k)| alpha_j * yj * k)
                    .sum();

                let margin = yi * (sum + self.b);

                if margin < 1.0 {
                    // Mise à jour de alpha_i
                    self.alphas[i] += self.lr * (1.0 - margin);
                    self.alphas[i] = self.alphas[i].clamp(0.0, self.c);

                    // Mise à jour du biais
                    self.b += self.lr * yi;
                }
            }
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let k_vec = self.compute_kernel(x);
        let sum: f64 = self.alphas.iter()
            .zip(self.labels.iter())
            .zip(k_vec.iter())
            .map(|((&alpha_j, &yj), &k)| alpha_j * yj * k)
            .sum();

        let result = sum + self.b;
        if result >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
