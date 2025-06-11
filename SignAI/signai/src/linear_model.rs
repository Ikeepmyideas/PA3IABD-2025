use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};
use serde_json;

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl LinearModel {
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            epochs,
        }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<f64>, activation: Option<fn(f64) -> f64>, derivative: Option<fn(f64) -> f64>) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                let z = self.predict_raw(xi);
                let output = match activation {
                    Some(activation_fn) => activation_fn(z),
                    None => z, 
                };
                let error = output - yi;

                // dérivée personnalisée pour tanh ou 1 pour linéaire
                let gradient = match derivative {
                    Some(deriv_fn) => deriv_fn(output) * error,
                    None => error,
                };

                for j in 0..self.weights.len() {
                    self.weights[j] -= self.learning_rate * gradient * xi[j]; //w = w - learning_rate*gradient 
                }
                self.bias -= self.learning_rate * gradient; //
            }
        }
    }

    pub fn predict_raw(&self, x: &Vec<f64>) -> f64 {
        self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias
    } // some de (wi*xi) + bias 

    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> f64 {
        let z = self.predict_raw(x);
        match activation {
            Some(f) => f(z),
            None => z,
        }
    }
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}

// Fonctions d'activation tanh pour la classification binaire
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(output: f64) -> f64 {
    1.0 - output.powi(2)
}


// Fonctions d'activation sigmoide pour la classification multiclasses
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

pub fn evaluate_accuracy(
    model: &LinearModel,
    X_test: &Vec<Vec<f64>>,
    y_test: &Vec<f64>,
    activation: Option<fn(f64) -> f64>,
) -> f64 {
    let mut correct = 0;

    for (x, &y_true) in X_test.iter().zip(y_test.iter()) {
        let y_pred = model.predict(x, activation);

        let y_pred_label = if y_pred >= 0.5 { 1.0 } else { 0.0 };

        if (y_pred_label - y_true).abs() < 1e-6 {
            correct += 1;
        }
    }

    correct as f64 / y_test.len() as f64
}