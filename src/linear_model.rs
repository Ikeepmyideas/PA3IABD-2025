use rand::Rng;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug)]
pub enum ActivationFn {
    Sign,
    Tanh,
    Linear,
}

#[derive(Debug)]
pub struct LinearModel {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
    pub max_epochs: usize,
    pub activation: ActivationFn,
}

impl LinearModel {
    pub fn new(n_features: usize, learning_rate: f32, max_epochs: usize, activation: ActivationFn) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        Self {
            weights: (0..n_features).map(|_| rng.gen_range(-1.0..=1.0)).collect(),  // inclut 1.0
            bias: rng.gen_range(-1.0..=1.0),
            learning_rate,
            max_epochs,
            activation,
        }
    }

    fn activate(&self, z: f32) -> f32 {
        match self.activation {
            ActivationFn::Sign => if z >= 0.0 { 1.0 } else { -1.0 },
            ActivationFn::Tanh => z.tanh(),
            ActivationFn::Linear => z,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f32>], y: &[f32], show_logs: bool) {
        for epoch in 0..self.max_epochs {
            let mut errors = 0;
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let y_pred = self.activate(self.raw_output(xi));

                if yi * y_pred <= 0.0 {
                    self.update_weights(xi, yi);
                    errors += 1;
                }
            }

            if show_logs {
                println!(
                    "Epoch {}: erreurs = {}, poids = {:?}, biais = {:.3}",
                    epoch,
                    errors,
                    self.weights.iter().map(|w| format!("{:.3}", w)).collect::<Vec<_>>(),
                    self.bias
                );
            }

            if errors == 0 {
                if show_logs {
                    println!("Convergence atteinte à l'epoch {}", epoch);
                }
                break;
            }
        }
    }

    fn update_weights(&mut self, xi: &[f32], yi: f32) {
        for (w, &xj) in self.weights.iter_mut().zip(xi.iter()) {
            *w += self.learning_rate * yi * xj;
        }
        self.bias += self.learning_rate * yi;
    }

    pub fn raw_output(&self, xi: &[f32]) -> f32 {
        self.weights
            .iter()
            .zip(xi.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>() + self.bias
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter()
            .map(|xi| self.activate(self.raw_output(xi)))
            .collect()
    }

    pub fn score_raw(&self, xi: &[f32]) -> f32 {
        self.raw_output(xi)
    }
}

#[derive(Debug)]
pub struct MultiClassLinear {
    pub models: Vec<LinearModel>,
}

impl MultiClassLinear {
    pub fn new(n_classes: usize, n_features: usize, learning_rate: f32, max_epochs: usize, activation: ActivationFn) -> Self {
        let models = (0..n_classes)
            .map(|_| LinearModel::new(n_features, learning_rate, max_epochs, activation))
            .collect();
        Self { models }
    }

    pub fn fit(&mut self, x: &[Vec<f32>], y: &[usize], show_logs: bool) {
        for (class_idx, model) in self.models.iter_mut().enumerate() {
            let y_binary: Vec<f32> = y.iter()
                .map(|&yi| if yi == class_idx { 1.0 } else { -1.0 })
                .collect();
            model.fit(x, &y_binary, show_logs);
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<usize> {
        x.iter()
            .map(|xi| {
                self.models
                    .iter()
                    .enumerate()
                    .map(|(idx, model)| (idx, model.score_raw(xi)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("aucun modèle trouvé")
                    .0
            })
            .collect()
    }
}
