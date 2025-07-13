use rand::Rng;

pub struct MLP {
    pub weights_hidden: Vec<Vec<f64>>,
    pub bias_hidden: Vec<f64>,
    pub weights_output: Vec<f64>,
    pub bias_output: f64,
    pub learning_rate: f64,
    pub epochs: usize,
    pub is_regression: bool,
    pub use_activation: bool,
}
pub struct MLPClassifier {
    pub weights_hidden: Vec<Vec<f64>>,
    pub bias_hidden: Vec<f64>,
    pub weights_output: Vec<Vec<f64>>, // chaque ligne = sortie pour une classe
    pub bias_output: Vec<f64>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
}

impl MLP {
    pub fn new(n_inputs: usize, n_hidden: usize, learning_rate: f64, epochs: usize,is_regression: bool,use_activation: bool) -> Self {
        let mut rng = rand::thread_rng();

        let weights_hidden = (0..n_hidden)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        println!("Poids initiaux cachés : {:?}", weights_hidden);

        let bias_hidden = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let weights_output = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = rng.gen_range(-1.0..1.0);

        Self {
            weights_hidden,
            bias_hidden,
            weights_output,
            bias_output,
            learning_rate,
            epochs,
            is_regression,
            use_activation, 
        }
    }

    fn activate(&self, x: f64) -> f64 {
        if self.use_activation {
            x.tanh()
        } else {
            x
        }
    }

    fn activate_derivative(&self, x: f64) -> f64 {
        if self.use_activation {
            1.0 - x.tanh().powi(2)
        } else {
            1.0
        }
    }


    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<f64>) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                // Forward pass
                let hidden_input: Vec<f64> = self.weights_hidden.iter()
                    .zip(self.bias_hidden.iter())
                    .map(|(w, &b)| w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                    .collect();

                let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

                let output_input = hidden_output.iter().zip(self.weights_output.iter()).map(|(h, w)| h * w).sum::<f64>() + self.bias_output;
                let output = if self.is_regression {
                    output_input
                } else {
                    self.activate(output_input)
                };

                // Backward pass
                let error = output - yi;
                let delta_output = if self.is_regression {
                    error // dérivée de l'identité = 1
                } else {
                    error * self.activate_derivative(output_input)
                };

                for i in 0..self.weights_output.len() {
                    self.weights_output[i] -= self.learning_rate * delta_output * hidden_output[i];
                }
                self.bias_output -= self.learning_rate * delta_output;

                for j in 0..self.weights_hidden.len() {
                    let delta_hidden = delta_output * self.weights_output[j] * self.activate_derivative(hidden_input[j]);
                    for k in 0..self.weights_hidden[j].len() {
                        self.weights_hidden[j][k] -= self.learning_rate * delta_hidden * xi[k];
                    }
                    self.bias_hidden[j] -= self.learning_rate * delta_hidden;
                }
            }
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let hidden_input: Vec<f64> = self.weights_hidden.iter()
            .zip(self.bias_hidden.iter())
            .map(|(w, &b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

        let output_input = hidden_output.iter()
            .zip(self.weights_output.iter())
            .map(|(h, w)| h * w)
            .sum::<f64>() + self.bias_output;

        if self.is_regression {
            output_input 
        } else {
            self.activate(output_input) // tanh sinon
        }
    }

}
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}
impl MLPClassifier {
    pub fn new(n_inputs: usize, n_hidden: usize, n_classes: usize, learning_rate: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights_hidden = (0..n_hidden)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let bias_hidden = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let weights_output = (0..n_classes)
            .map(|_| (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let bias_output = (0..n_classes).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            weights_hidden,
            bias_hidden,
            weights_output,
            bias_output,
            learning_rate,
            epochs,
            n_classes,
        }
    }

    fn activate(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn activate_derivative(&self, x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<usize>) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                // Forward hidden
                let hidden_input: Vec<f64> = self.weights_hidden.iter()
                    .zip(self.bias_hidden.iter())
                    .map(|(w, &b)| w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                    .collect();

                let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

                // Forward output
                let logits: Vec<f64> = self.weights_output.iter()
                    .zip(self.bias_output.iter())
                    .map(|(w_out, &b)| hidden_output.iter().zip(w_out.iter()).map(|(h, w)| h * w).sum::<f64>() + b)
                    .collect();

                let probs = softmax(&logits);

                // Backward output
                let mut delta_output = vec![0.0; self.n_classes];
                for j in 0..self.n_classes {
                    delta_output[j] = probs[j] - if j == yi { 1.0 } else { 0.0 };
                }

                // Gradient pour poids de sortie
                for j in 0..self.n_classes {
                    for k in 0..self.weights_output[j].len() {
                        self.weights_output[j][k] -= self.learning_rate * delta_output[j] * hidden_output[k];
                    }
                    self.bias_output[j] -= self.learning_rate * delta_output[j];
                }

                // Backpropagation dans la couche cachée
                for h in 0..self.weights_hidden.len() {
                    let mut grad = 0.0;
                    for j in 0..self.n_classes {
                        grad += delta_output[j] * self.weights_output[j][h];
                    }
                    let delta_h = grad * self.activate_derivative(hidden_input[h]);

                    for i in 0..self.weights_hidden[h].len() {
                        self.weights_hidden[h][i] -= self.learning_rate * delta_h * xi[i];
                    }
                    self.bias_hidden[h] -= self.learning_rate * delta_h;
                }
            }
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> usize {
        let hidden_input: Vec<f64> = self.weights_hidden.iter()
            .zip(self.bias_hidden.iter())
            .map(|(w, &b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

        let logits: Vec<f64> = self.weights_output.iter()
            .zip(self.bias_output.iter())
            .map(|(w_out, &b)| hidden_output.iter().zip(w_out.iter()).map(|(h, w)| h * w).sum::<f64>() + b)
            .collect();

        let probs = softmax(&logits);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}
