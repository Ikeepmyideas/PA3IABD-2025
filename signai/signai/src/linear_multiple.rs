use std::collections::HashMap;
use csv::Reader;
mod prepare_dataset;
use crate::prepare_dataset::DataPoint;

pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
}

impl LogisticRegression {
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        LogisticRegression {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            epochs,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<u8>) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                let linear_output = Self::dot_product(&self.weights, xi) + self.bias;
                let prediction = Self::sigmoid(linear_output);
                let error = prediction - yi as f64;

                for j in 0..self.weights.len() {
                    self.weights[j] -= self.learning_rate * error * xi[j];
                }

                self.bias -= self.learning_rate * error;
            }
        }
    }

    pub fn predict_single(&self, x: &Vec<f64>) -> f64 {
        Self::dot_product(&self.weights, x) + self.bias
    }
    pub fn predict_multiclass(models: &Vec<LogisticRegression>, x: &Vec<f64>) -> usize {
    let mut best_class = 0;
    let mut best_score = f64::MIN;

    for (i, model) in models.iter().enumerate() {
        let score = model.predict_single(x);
        let proba = 1.0 / (1.0 + (-score).exp());
        if proba > best_score {
            best_score = proba;
            best_class = i;
        }
    }

    best_class
}

}

pub fn train_one_vs_all(
    data: Vec<DataPoint>,
    n_classes: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
) -> Vec<LogisticRegression> {
    let mut X: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<usize> = Vec::new();

    for dp in data {
        X.push(dp.features);
        y.push(dp.label);
    }

    let mut models = Vec::new();
    for class_id in 0..n_classes {
        let y_binary: Vec<u8> = y.iter().map(|&y| if y == class_id { 1 } else { 0 }).collect();
        let mut model = LogisticRegression::new(n_features, learning_rate, epochs);
        model.fit(&X, &y_binary);
        models.push(model);
        println!("Modèle pour classe {} entraîné", class_id);
    }

    models
}
