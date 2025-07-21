// Structures principales :
// - SVMClassifierRBF : SVM binaire avec noyau RBF (gaussien).
// - SVMMultiClassRBF : SVM multiclasse basé sur plusieurs classifieurs binaires OvA.
// Le SVM binaire distingue deux classes : +1 et -1.

use crate::loss::mse;
use nalgebra::DVector;

// === SVM binaire avec noyau RBF ===
pub struct SVMClassifierRBF {
    pub loss_history: Vec<f32>,
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
            loss_history: Vec::new(),
        }
    }

    fn rbf(&self, x: &[f64], c: &[f64]) -> f64 {
        x.iter()
            .zip(c.iter())
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum::<f64>()
            .mul_add(-self.gamma, 0.0)
            .exp()
    }

    pub fn compute_kernel(&self, x: &[f64]) -> Vec<f64> {
        self.support_vectors
            .iter()
            .map(|sv| self.rbf(x, sv))
            .collect()
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.support_vectors = x.to_vec();
        self.labels = y.to_vec();
        self.alphas = DVector::zeros(x.len());
        self.loss_history.clear();

        for _ in 0..self.epochs {
            for i in 0..x.len() {
                let xi = &x[i];
                let yi = y[i];

                let k_vec = self.compute_kernel(xi);
                let sum: f64 = self
                    .alphas
                    .iter()
                    .zip(self.labels.iter())
                    .zip(k_vec.iter())
                    .map(|((&alpha_j, &yj), &k)| alpha_j * yj * k)
                    .sum();

                let margin = yi * (sum + self.b);

                if margin < 1.0 {
                    self.alphas[i] += self.lr * (1.0 - margin);
                    self.alphas[i] = self.alphas[i].clamp(0.0, self.c);
                    self.b += self.lr * yi;
                }
            }

            // === Calcul de la loss à chaque époque ===
            let preds: Vec<f32> = x.iter().map(|xi| self.predict(xi) as f32).collect();
            let targets: Vec<f32> = y.iter().map(|&v| v as f32).collect();

            if let Some(loss) = mse(&preds, &targets) {
                self.loss_history.push(loss);
            }
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let k_vec = self.compute_kernel(x);
        let sum: f64 = self
            .alphas
            .iter()
            .zip(self.labels.iter())
            .zip(k_vec.iter())
            .map(|((&alpha_j, &yj), &k)| alpha_j * yj * k)
            .sum();

        if sum + self.b >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn decision_function(&self, x: &[f64]) -> f64 {
        let k_vec = self.compute_kernel(x);
        self.alphas
            .iter()
            .zip(self.labels.iter())
            .zip(k_vec.iter())
            .map(|((&alpha_j, &yj), &k)| alpha_j * yj * k)
            .sum::<f64>()
            + self.b
    }
}

// === SVM Multiclasse OvA (One-vs-All) ===
// Utilise plusieurs classifieurs binaires RBF (un par classe)

pub struct SVMMultiClassRBF {
    pub classifiers: Vec<SVMClassifierRBF>, 
    pub class_labels: Vec<usize>,
    pub gamma: f64,
    pub c: f64,
    pub lr: f64,
    pub epochs: usize,
}

impl SVMMultiClassRBF {
    pub fn new(gamma: f64, c: f64, lr: f64, epochs: usize) -> Self {
        Self {
            classifiers: Vec::new(),
            class_labels: Vec::new(),
            gamma,
            c,
            lr,
            epochs,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        let mut unique_labels = y.to_vec();
        unique_labels.sort_unstable();
        unique_labels.dedup();

        for &label in &unique_labels {
            let binary_labels: Vec<f64> = y
                .iter()
                .map(|&yi| if yi == label { 1.0 } else { -1.0 })
                .collect();

            let mut clf = SVMClassifierRBF::new(self.gamma, self.c, self.lr, self.epochs);
            clf.fit(x, &binary_labels);

            self.classifiers.push(clf);
            self.class_labels.push(label);
        }
    }

    pub fn predict(&self, x: &[f64]) -> usize {
        self.classifiers
            .iter()
            .zip(self.class_labels.iter())
            .map(|(clf, &label)| (clf.decision_function(x), label))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, label)| label)
            .unwrap_or(self.class_labels[0])
    }

    pub fn get_loss_history(&self) -> Vec<f32> {
        if self.classifiers.is_empty() {
            return vec![0.0; self.epochs];
        }

        let mut total_loss = vec![0.0; self.epochs];
        let mut counts = vec![0; self.epochs];

        for clf in &self.classifiers {
            for (i, &loss) in clf.loss_history.iter().enumerate() {
                if i < self.epochs {
                    total_loss[i] += loss;
                    counts[i] += 1;
                }
            }
        }

        total_loss
            .iter()
            .zip(counts.iter())
            .map(|(&loss, &count)| if count > 0 { loss / count as f32 } else { 0.0 })
            .collect()
    }
}
