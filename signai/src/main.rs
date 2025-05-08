use ndarray::Array1;

mod extract_features; 
use extract_features::extract_features; 

mod pmc;
use pmc::PMC;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn predict(input: &Array1<f64>, weights: &Array1<f64>, bias: f64) -> f64 {
    let z = input.dot(weights) + bias;
    sigmoid(z)
}

fn log_loss(y_true: f64, y_pred: f64) -> f64 {
    //ajouter un petit epsilon pour éviter log(0)
    let e = 1e-15;
    let y_pred_clipped = y_pred.max(e).min(1.0 - e);

    - (y_true * y_pred_clipped.ln() + (1.0 - y_true) * (1.0 - y_pred_clipped).ln())
}

fn main() {
    use ndarray::Array1;

    // === Données d'apprentissage ===
    let test_data = vec![
        (Array1::from(vec![0.5, 1.2, 0.5f64.powi(2), 1.2f64.powi(2), 0.5 * 1.2]), 1.0),
        (Array1::from(vec![0.3, -0.7, 0.3f64.powi(2), (-0.7f64).powi(2), 0.3 * -0.7]), 0.0),
        (Array1::from(vec![1.0, 0.5, 1.0f64.powi(2), 0.5f64.powi(2), 1.0 * 0.5]), 1.0),
        (Array1::from(vec![-0.2, -1.0, (-0.2f64).powi(2), (-1.0f64).powi(2), -0.2 * -1.0]), 0.0),
    ];

    let learning_rate = 0.1;
    let epochs = 100;

    // === Entraînement du modèle linéaire ===
    let mut weights = Array1::from(vec![0.0; 5]);
    let mut bias = 0.0;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut gradient_w: Array1<f64> = Array1::zeros(weights.len());
        let mut gradient_b = 0.0;

        for (x, y_true) in &test_data {
            let y_pred = predict(x, &weights, bias);
            let loss = log_loss(*y_true, y_pred);
            total_loss += loss;

            let error = y_pred - y_true;
            gradient_w = gradient_w + &(x * error);
            gradient_b += error;
        }

        let n = test_data.len() as f64;
        weights = &weights - &gradient_w.mapv(|g| learning_rate * g / n);
        bias -= learning_rate * gradient_b / n;
        println!("Epoch {:3} | Loss moyenne : {:.5}", epoch + 1, total_loss / n);
    }

    println!("\nPoids finaux (linéaire) : {:?}", weights);
    println!("Biais final (linéaire)  : {:.4}", bias);

    // === Entraînement du PMC ===
    let mut pmc = PMC::new(5, 3, 0.1);
    let test_data_vec: Vec<(Vec<f64>, f64)> = test_data
        .iter()
        .map(|(x, y)| (x.to_vec(), *y))
        .collect();
    pmc.train(&test_data_vec, 100);

    // === Test d'une image avec les deux modèles ===
    let image_path = "../dataset/train/B/B_001.jpg";
    match extract_features(image_path) {
        Ok((white, aspect)) => {
            let features_vec = vec![
                white,
                aspect,
                white.powi(2),
                aspect.powi(2),
                white * aspect,
            ];
            let features_array = Array1::from(features_vec.clone());

            // Prédiction modèle linéaire
            let y_pred_linear = predict(&features_array, &weights, bias);
            println!("\n=== Prédiction modèle linéaire ===");
            println!("Image : {}", image_path);
            println!("  Ratio de pixels blancs : {:.4}", white);
            println!("  Aspect ratio            : {:.4}", aspect);
            println!("  Probabilité classe 1    : {:.4}", y_pred_linear);
            println!("  Classe prédite          : {}", if y_pred_linear >= 0.5 { 1 } else { 0 });

            // Prédiction réseau PMC
            let y_pred_pmc = pmc.predict(&features_vec);
            println!("\n=== Prédiction réseau PMC ===");
            println!("Image : {}", image_path);
            println!("  Probabilité classe 1    : {:.4}", y_pred_pmc);
            println!("  Classe prédite          : {}", if y_pred_pmc >= 0.5 { 1 } else { 0 });
        }
        Err(e) => println!("Erreur lors du traitement de l'image : {}", e),
    }
}

