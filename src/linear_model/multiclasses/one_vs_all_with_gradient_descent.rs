
use crate::linear_model::linear_gradient_descent::LinearModel;

/// One-vs-All wrapper pour classification multiclasse
pub struct OneVsAllModel {
    pub models: Vec<LinearModel>,  // Un modèle par classe
    pub n_classes: usize,
}

impl OneVsAllModel {
    pub fn new(n_classes: usize, n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        let models = (0..n_classes)
            .map(|_| LinearModel::new(n_features, learning_rate, epochs))
            .collect();
        Self { models, n_classes }
    }

    pub fn fit(
        &mut self,
        X: &Vec<Vec<f64>>,
        y: &Vec<usize>,  // étiquettes 0..K-1
        activation: Option<fn(f64) -> f64>,
        derivative: Option<fn(f64) -> f64>,
    ) {
        for (k, model) in self.models.iter_mut().enumerate() {
            // Créer une cible binaire : 1 si classe == k, sinon 0
            let binary_targets: Vec<f64> = y.iter().map(|&yi| if yi == k { 1.0 } else { 0.0 }).collect();
            model.fit(X, &binary_targets, activation, derivative);
        }
    }

    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> usize {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_class = 0;

        for (k, model) in self.models.iter().enumerate() {
            let score = model.predict(x, activation);
            if score > best_score {
                best_score = score;
                best_class = k;
            }
        }

        best_class
    }
}
