use crate::linear_model::linear_pseudo_inverse::PseudoInverseModel;
use crate::linear_model::utils::activations::{sigmoid, tanh, relu};

pub struct OneVsAllPseudoInverse {
    pub models: Vec<Option<PseudoInverseModel>>,
    pub n_classes: usize,
}

impl OneVsAllPseudoInverse {
    pub fn new(n_classes: usize) -> Self {
        Self {
            models: vec![None; n_classes],
            n_classes,
        }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<usize>) {
        for k in 0..self.n_classes {
            let binary_targets: Vec<f64> = y.iter().map(|&yi| if yi == k { 1.0 } else { 0.0 }).collect();
            self.models[k] = PseudoInverseModel::fit(X, &binary_targets).ok();
        }
    }

    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> usize {
    self.models
        .iter()
        .enumerate()
        .filter_map(|(k, model)| {
            model.as_ref().and_then(|m| {
                match m.predict(x, activation) {
                    Ok(score) => Some((k, score)),
                    Err(e) => {
                        eprintln!("[ERREUR - predict OneVsAll] Classe {k} : {e}");
                        None
                    }
                }
            })
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k)
        .unwrap_or(0)
}

}
