use ndarray::Array1;
mod extract_features; 
use extract_features::extract_features; 

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
    //let weights = Array1::from(vec![0.4, -0.4]);
    //let bias = 0.1;

    // Données d'entrée (features) + labels réels
    /*
    let test_data = vec![
        (Array1::from(vec![0.5, 1.2]), 1.0),  // A
        (Array1::from(vec![0.3, -0.7]), 0.0), // pas A
        (Array1::from(vec![1.0, 0.5]), 1.0),  //  A
        (Array1::from(vec![-0.2, -1.0]), 0.0),// pas A
    ];
    */
    let test_data = vec![
        (Array1::from(vec![0.5, 1.2, 0.5f64.powi(2), 1.2f64.powi(2), 0.5 * 1.2]), 1.0),
        (Array1::from(vec![0.3, -0.7, 0.3f64.powi(2), (-0.7f64).powi(2), 0.3 * -0.7]), 0.0),
        (Array1::from(vec![1.0, 0.5, 1.0f64.powi(2), 0.5f64.powi(2), 1.0 * 0.5]), 1.0),
        (Array1::from(vec![-0.2, -1.0, (-0.2f64).powi(2), (-1.0f64).powi(2), -0.2 * -1.0]), 0.0),
    ];

    let learning_rate = 0.1;
    let epochs = 100;

    // poinds er biais 
    //let mut weights = Array1::from(vec![0.0, 0.0]);
    let mut weights = Array1::from(vec![0.0; 5]);

    let mut bias = 0.0;

    for epoch in 0..epochs {
        let mut total_loss = 0.0; // perte cumulée 

        let mut gradient_w = Array1::zeros(weights.len()); 
        let mut gradient_b = 0.0;
        // prediction + calcule la perte + accumuler les gradients 
        for(x,y_true) in &test_data{
            let y_pred = predict(x,&weights, bias);
            let loss = log_loss(*y_true, y_pred);
            total_loss += loss;

            let error = y_pred -y_true;
            gradient_w = gradient_w + &(x * error);
            gradient_b += error; 
        }
        // moyenne des gradients et mise a jour des poids 
        let n = test_data.len() as f64;
        weights = &weights - &gradient_w.mapv(|g: f64| -> f64 { learning_rate * g / n });
        bias -= learning_rate * gradient_b / n;
        println!("Epoch {:3} | Loss moyenne : {:.5}", epoch + 1, total_loss / n);
    }
    println!("\nPoids finaux : {:?}", weights);
    println!("Biais final : {:.4}", bias);

    println!("\n=== Prédictions finales sur les données ===");
    /*
    for (i, (x, y_true)) in test_data.iter().enumerate() {
        let y_pred = predict(x, &weights, bias);
        let prediction = if y_pred >= 0.5 { 1.0 } else { 0.0 }; // seuil de décision
        let erreur = (y_true - y_pred).abs();
        let loss = log_loss(*y_true, y_pred);

        println!("Exemple {} :", i + 1);
        println!("  Entrée         : {:?}", x);
        println!("  Vraie classe   : {}", y_true);
        println!("  Probabilité    : {:.4}", y_pred);
        println!("  Prédiction     : {}", prediction);
        println!("  Erreur absolue : {:.4}", erreur);
        println!("  Log-loss       : {:.4}", loss);
        println!();
    }
    */

    let image_path = "../dataset/train/B/B_001.jpg";
    match extract_features(image_path){
        Ok((white,aspect)) => {
            //let x = Array1::from(vec![white, aspect]);
            let x = Array1::from(vec![
                white,
                aspect,
                white.powi(2),
                aspect.powi(2),
                white * aspect
            ]);            
            let y_pred = predict(&x, &weights, bias);
            println!("Image : {}", image_path);
            println!("  Ratio de pixels blancs : {:.4}", white);
            println!("  Aspect ratio            : {:.4}", aspect);
            println!("  Probabilité classe 1    : {:.4}", y_pred);
            println!("  Classe prédite          : {}", if y_pred >= 0.5 { 1 } else { 0 });
        }
        Err(e) => println!("Erreur lors du traitement de l'image : {}", e),
    }
   
    }
