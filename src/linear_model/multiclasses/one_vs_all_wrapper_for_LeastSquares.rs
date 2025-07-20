//Le modèle LeastSquaresModel est linéaire, sans activation à l'entraînement.
//On lui applique une activation non linéaire à la prédiction (sigmoid, tanh, ReLU).

use crate::linear_model::linear_least_squares::LeastSquaresModel;
use crate::linear_model::utils::activations::{sigmoid, tanh, relu};

pub struct OneVsAllLeastSquares {
    pub models: Vec<LeastSquaresModel>,
    pub n_classes: usize,
}

impl OneVsAllLeastSquares {
    pub fn new(n_classes: usize, n_features: usize) -> Self {
        let models = (0..n_classes)
            .map(|_| LeastSquaresModel {
                weights: vec![0.0; n_features],
                bias: 0.0,
            })
            .collect();
        Self { models, n_classes }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<usize>) {
        for (k, model) in self.models.iter_mut().enumerate() {
            let binary_targets: Vec<f64> = y.iter().map(|&yi| if yi == k { 1.0 } else { 0.0 }).collect();
            *model = LeastSquaresModel::fit(X, &binary_targets);
        }
    }

    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> usize {
        self.models
            .iter()
            .enumerate()
            .map(|(k, model)| (k, model.predict(x, activation)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(k, _)| k)
            .unwrap_or(0)
    }
}
