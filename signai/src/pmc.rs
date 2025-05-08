use rand::Rng;// genere les poids aléatoires 
// structure du réseau PMC 
// Vec<f64> est un vecteur (liste dynamique) de nombres à virgule flottante
// vec! est un macro qui permet de créer facilement un vecteur (Vec) dynamique
pub struct PMC {
    input_size: usize, // nb de caractéristiques 5
    hidden_size: usize, // nb de neurones dans la couche cachée 
    w1: Vec<Vec<f64>>, // poids entre entrée et cachée 
    b1: Vec<f64>, // biais pour chaque neurone caché 
    w2: Vec<f64>, // poids entre cachée et sortie 
    b2: f64, // biais pour la sortie 
    learning_rate: f64, // vitesse d apprentissage 
}
impl PMC {
    // creation du PMC avec initialisation aléatoire des poids 
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        //creer une matrice de taille hidden_size x input_size
        let w1 = (0..hidden_size).map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();        
        let b1 = vec![0.0; hidden_size]; // vecteur de zero de taille hidden_size
        let w2 = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
       
        // initialiser une nouvelle instance 

        Self {
            input_size,
            hidden_size,
            w1,
            b1,
            w2,
            b2: 0.0,
            learning_rate,
        }
    }
    // fonction d activation , retourne x si x>0 sinon 0 
    fn relu(x: f64) -> f64{
        x.max(0.0)
    }
    /*
    fn relu_deriv(x: f64) -> f64 {
        if x > 0.0 { 1.0} else {0.0}
    }
        */
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    /*calcul des cativations de la couche cachée 
      self.w1.iter() -> parcourt chaque ligne de la matrice de poids w1 (pour chaque neurone caché)
      .zip(&self.b1) -> associe chaque poids w à son biais b
      .map -> pour chaque neurone caché
    */
    pub fn predict(&self, input: &[f64]) -> f64 {
        // Calcule les activations des neurones cachés
        let hidden: Vec<f64> = self.w1.iter().zip(&self.b1).map(|(weights, bias)| {
            let sum: f64 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
            PMC::relu(sum + bias)
        }).collect();

        // Calcule la sortie finale avec sigmoïde
        let output_sum: f64 = self.w2.iter().zip(&hidden).map(|(w, h)| w * h).sum();
        PMC::sigmoid(output_sum + self.b2)
    }
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
    
            for (x, y_true) in data {
                // Étape 1 : propagation avant
                // couche cachée
                let hidden_input: Vec<f64> = self.w1.iter().zip(&self.b1).map(|(w, b)| {
                    let sum: f64 = w.iter().zip(x).map(|(wi, xi)| wi * xi).sum();
                    sum + b
                }).collect();
    
                let hidden_output: Vec<f64> = hidden_input.iter().map(|z| PMC::relu(*z)).collect();
    
                // sortie finale
                let z2: f64 = self.w2.iter().zip(&hidden_output).map(|(w, h)| w * h).sum::<f64>() + self.b2;
                let y_pred = PMC::sigmoid(z2);
    
                // Calcul de la perte
                let loss = -(*y_true * y_pred.ln() + (1.0 - *y_true) * (1.0 - y_pred).ln());
                total_loss += loss;
    
                // Étape 2 : rétropropagation
                let erreur_sortie = y_pred - y_true;
    
                for i in 0..self.hidden_size {
                    // dérivée ReLU pour chaque neurone caché
                    let deriv = if hidden_input[i] > 0.0 { 1.0 } else { 0.0 };
    
                    for j in 0..self.input_size {
                        // correction des poids entre entrée et cachée
                        self.w1[i][j] -= self.learning_rate * erreur_sortie * self.w2[i] * deriv * x[j];
                    }
    
                    // correction du biais caché
                    self.b1[i] -= self.learning_rate * erreur_sortie * self.w2[i] * deriv;
    
                    // correction des poids entre cachée et sortie
                    self.w2[i] -= self.learning_rate * erreur_sortie * hidden_output[i];
                }
    
                // correction du biais de sortie
                self.b2 -= self.learning_rate * erreur_sortie;
            }
    
            println!("Époque {:3} | Perte moyenne : {:.4}", epoch + 1, total_loss / data.len() as f64);
        }
    }
    
}
