use crate::linear_model::utils::activations::softmax;
use rand::Rng;

pub struct SoftmaxModel {
    pub weights: Vec<Vec<f64>>,  // [n_classes][n_features]
    pub biases: Vec<f64>,        // [n_classes]
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
    pub lambda: f64,             // Coefficient de régularisation L2
}

impl SoftmaxModel {
    pub fn new(n_features: usize, n_classes: usize, learning_rate: f64, epochs: usize, lambda: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..n_classes)
            .map(|_| (0..n_features).map(|_| rng.gen_range(-0.01..0.01)).collect())
            .collect();

        Self {
            weights,
            biases: vec![0.0; n_classes],
            learning_rate,
            epochs,
            n_classes,
            lambda,
        }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<usize>) {
    let n = X.len();
    let d = X[0].len();

    assert_eq!(y.len(), n, "Longueur de y différente de X");
    assert_eq!(self.weights.len(), self.n_classes, "Poids mal initialisés");
    assert_eq!(self.biases.len(), self.n_classes);

    for row in X {
        assert_eq!(row.len(), d, "Incohérence dans les dimensions de X");
    }

    for w in &self.weights {
        assert_eq!(w.len(), d, "Incohérence dans les dimensions des poids");
    }

    println!(" Démarrage de l'entraînement Softmax : {} epochs", self.epochs);

    for epoch in 0..self.epochs {
        let mut total_loss = 0.0;

        for i in 0..n {
            let xi = &X[i];
            let yi = y[i];
            assert!(yi < self.n_classes, "Classe {yi} hors des bornes !");

            // Calcul des scores
            let mut scores = vec![0.0; self.n_classes];
            for k in 0..self.n_classes {
                scores[k] = self.weights[k]
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.biases[k];
            }

            // Softmax
            let probs = crate::linear_model::utils::activations::softmax(&scores);
            if probs.iter().any(|p| !p.is_finite()) {
                panic!(" Probas invalides : {:?}", probs);
            }

            total_loss -= probs[yi].ln();

            // Mise à jour
            for k in 0..self.n_classes {
                let error = probs[k] - if k == yi { 1.0 } else { 0.0 };

                for j in 0..d {
                    //Nouvelle perte = erreur sur les donnees + λ ×somme des poids au carre
                    self.weights[k][j] -= self.learning_rate
                        * (error * xi[j] + self.lambda * self.weights[k][j]); // régularisation pour éviter l'overfitting si les poids deviennet trop grands 
                }

                self.biases[k] -= self.learning_rate * error;
            }
        }

        if epoch % 100 == 0 {
            println!(" Epoch {}/{} — Loss moyenne: {:.4}", epoch + 1, self.epochs, total_loss / n as f64);
        }
    }

    println!(" Entraînement terminé !");
}


    pub fn predict(&self, x: &Vec<f64>) -> usize {
        let probs = self.predict_proba(x);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn predict_proba(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut scores = vec![0.0; self.n_classes];
        for k in 0..self.n_classes {
            scores[k] = self.weights[k]
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum::<f64>()
                + self.biases[k];
        }

        softmax(&scores)
    }
}
