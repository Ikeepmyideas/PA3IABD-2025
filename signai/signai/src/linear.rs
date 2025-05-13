// src/linear.rs

use ndarray::Array1;

/// Modèle linéaire y = w · x + b
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
        }
    }

    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        self.weights.dot(input) + self.bias
    }

    pub fn train(&mut self, data: &Vec<(Array1<f64>, f64)>, learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (x, y_true) in data {
                let y_pred = self.predict(x);
                let error = y_pred - y_true;

                // Mise à jour des poids
                self.weights = self.weights.clone() - &(x * learning_rate * error);
                self.bias -= learning_rate * error;

                total_loss += error.powi(2);
            }

            println!(
                "Époque {:>2} | Perte moyenne : {:.5}",
                epoch + 1,
                total_loss / data.len() as f64
            );
        }
    }
}
