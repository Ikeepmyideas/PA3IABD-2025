use nalgebra::{DMatrix, DVector};

#[derive(PartialEq, Debug)]
pub enum RBFMode {
    Regression,
    BinaryClassification,
    MultiClassification(usize),
}

pub struct RBFN {
    pub centers: Vec<Vec<f64>>,     // Chaque point devient un centre
    pub sigma: f64,                 // écart-type de la gaussienne
    pub weights: DMatrix<f64>,     // (n_hidden, n_outputs)
    pub learning_rate: f64,
    pub epochs: usize,
    pub mode: RBFMode,
    pub train_losses: Vec<f64>,
    pub test_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>,
    pub test_accuracies: Vec<f64>,
}

impl RBFN {
    pub fn new(sigma: f64, learning_rate: f64, epochs: usize, mode: RBFMode) -> Self {
        RBFN {
            centers: Vec::new(), // remplis plus tard par x.clone()
            sigma,
            weights: DMatrix::zeros(0, 0), // taille définie plus tard
            learning_rate,
            epochs,
            mode,
            train_losses: Vec::new(),
            test_losses: Vec::new(),
            train_accuracies: Vec::new(),
            test_accuracies: Vec::new(),
        }
    }

    fn gaussian(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-dist_sq / (2.0 * self.sigma.powi(2))).exp()
    }

    fn compute_phi(&self, x: &Vec<Vec<f64>>) -> DMatrix<f64> {
        let n_samples = x.len();
        let n_hidden = self.centers.len(); //(nombre de centres = nombre de neurones dans la couche RBF)
        let mut data = Vec::with_capacity(n_samples * n_hidden);//Φ∈R (n_samples×n_hidden)  === vecteur pour contenir toutes les valeurs φ(x, c)

        //Pour chaque exemple xi, on calcule son activation avec chaque centre cj via la fonction radiale gaussian.
        for xi in x {
            for cj in &self.centers {
                data.push(self.gaussian(xi, cj));
            }
        }

        //On transforme le vecteur data en une matrice dense Φ de dimension (n_samples × n_hidden).
        DMatrix::from_row_slice(n_samples, n_hidden, &data)
    }

    pub fn fit_closed_form(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        self.centers = x.clone();
        let phi = self.compute_phi(x);
        let phi_t = phi.transpose();
        let phi_t_phi = &phi_t * &phi;

        if let Some(inv) = phi_t_phi.try_inverse() {
            let y_matrix = DVector::from_vec(y.clone());
            let weights = inv * (phi_t * y_matrix);
            self.weights = DMatrix::from_columns(&[weights]);
        } else {
            eprintln!("Erreur : matrice non inversible");
        }
    }

    pub fn fit_rosenblatt_binary(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        assert_eq!(self.mode, RBFMode::BinaryClassification);
        self.centers = x.clone();  // chaque point devient un centre RBF
        let n_hidden = self.centers.len();
        self.weights = DMatrix::zeros(n_hidden, 1);
        // Calcul de la matrice Φ
        let phi = self.compute_phi(x);

        for _ in 0..self.epochs {
            for (i, phi_row) in phi.row_iter().enumerate() {
                let y_i = y[i];
                let prediction = phi_row.dot(&self.weights.column(0));

                if y_i * prediction <= 0.0 {
                    let correction = phi_row * (self.learning_rate * y_i);
                    let mut current_weights = self.weights.column_mut(0);
                    current_weights += correction.transpose();
                }
            }
        }
    }

    pub fn fit_gradient_descent(&mut self, x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>, x_test: &Vec<Vec<f64>>, y_test: &Vec<Vec<f64>>) {
        self.centers = x.clone();
        let n_hidden = self.centers.len();
        let n_outputs = match self.mode {
            RBFMode::MultiClassification(k) => k,
            _ => panic!("fit_gradient_descent ne doit être utilisé que pour la classification multiclasse"),
        };

        self.weights = DMatrix::zeros(n_hidden, n_outputs);
        let phi = self.compute_phi(x);

        for _ in 0..self.epochs {
            let prediction = &phi * &self.weights;
            let y_flat: Vec<f64> = y.iter().flat_map(|v| v.iter()).copied().collect();
            let y_matrix = DMatrix::from_row_slice(x.len(), n_outputs, &y_flat);
            let error = &y_matrix - &prediction;

            // Utilisation par emprunt (pas de move)
            let gradient = phi.transpose() * &error;

            self.weights += self.learning_rate * gradient / (x.len() as f64);

            // Utilisation par move ensuite (ok maintenant)
            let loss = error.map(|v| v.powi(2)).sum() / (x.len() as f64);
            self.train_losses.push(loss);


            // === Train accuracy ===
            let mut correct = 0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                let pred_label = self.predict_label(xi) as usize;
                let true_label = yi.iter().position(|&v| v == 1.0).unwrap_or(usize::MAX);
                if pred_label == true_label {
                    correct += 1;
                }
            }
            self.train_accuracies.push(correct as f64 / x.len() as f64);

            // === Test accuracy & loss ===
            let mut correct_test = 0;
            let mut test_loss = 0.0;
            for (xi, yi) in x_test.iter().zip(y_test.iter()) {
                let preds = self.predict(xi);
                let true_label = yi.iter().position(|&v| v == 1.0).unwrap_or(usize::MAX);
                let pred_label = preds
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(usize::MAX);
                if pred_label == true_label {
                    correct_test += 1;
                }
                let pred_vec = DVector::from_vec(preds);
                let true_vec = DVector::from_vec(yi.clone());
                test_loss += (&true_vec - &pred_vec).map(|v| v.powi(2)).sum();
            }
            self.test_accuracies.push(correct_test as f64 / x_test.len() as f64);
            self.test_losses.push(test_loss / x_test.len() as f64);
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> Vec<f64> {
        let phi: Vec<f64> = self.centers.iter().map(|c| self.gaussian(x, c)).collect();
        let phi_vec = DVector::from_vec(phi);
        let result = phi_vec.transpose() * &self.weights;
        result.row(0).iter().copied().collect()
    }

    pub fn predict_label(&self, x: &Vec<f64>) -> f64 {
        match self.mode {
            RBFMode::Regression => self.predict(x)[0],
            RBFMode::BinaryClassification => {
                let out = self.predict(x)[0];
                if out >= 0.0 { 1.0 } else { -1.0 }
            }
            RBFMode::MultiClassification(_) => {
                let output = self.predict(x);
                output.iter()
                      .enumerate()
                      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                      .map(|(i, _)| i as f64)
                      .unwrap_or(-1.0)
            }
        }
    }
}