// src/pmc.rs
use ndarray::{Array1, Array2};
use rand::Rng;

pub struct PMC {
    w1: Array2<f64>, // poids entrée -> cachée
    b1: Array1<f64>,
    w2: Array1<f64>, // poids cachée -> sortie
    b2: f64,
}

impl PMC {
    pub fn new(n_inputs: usize, n_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w1: Array2::from_shape_fn((n_hidden, n_inputs), |_| rng.gen_range(-1.0..1.0)),
            b1: Array1::zeros(n_hidden),
            w2: Array1::from((0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect::<Vec<_>>()),
            b2: 0.0,
        }
    }

    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        let hidden_input = self.w1.dot(input) + &self.b1;
        let hidden_output = Self::relu(&hidden_input);
        let output = self.w2.dot(&hidden_output) + self.b2;
        Self::sigmoid(output) // ou juste `output` si tu veux une sortie linéaire
    }
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, lr: f64, epochs: usize) {
    let n_samples = x.shape()[0];
    for _ in 0..epochs {
        for i in 0..n_samples {
            let input = x.row(i).to_owned();
            let target = y[i];

            // Forward pass
            let z1 = self.w1.dot(&input) + &self.b1;
            let a1 = Self::relu(&z1);
            let z2 = self.w2.dot(&a1) + self.b2;
            let y_pred = Self::sigmoid(z2);

            // Backward pass
            let error = y_pred - target;
            let d_output = error * y_pred * (1.0 - y_pred); // dérivée sigmoïde
            let d_w2 = &a1 * d_output;
            let d_b2 = d_output;

            let mut d_a1 = &self.w2 * d_output;
            for i in 0..d_a1.len() {
                if z1[i] <= 0.0 {
                    d_a1[i] = 0.0; // dérivée ReLU
                }
            }

            let d_w1 = d_a1
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&input.view().insert_axis(ndarray::Axis(0)));
            let d_b1 = d_a1;

            // Mise à jour des poids
            self.w2 = &self.w2 - &(lr * &d_w2);
            self.b2 -= lr * d_b2;
            self.w1 = &self.w1 - &(lr * d_w1);
            self.b1 = &self.b1 - &(lr * &d_b1);
        }
    }
}

}
