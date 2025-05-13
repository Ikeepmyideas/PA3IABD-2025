mod linear;
mod prepare_dataset;
mod pmc;

use linear::LinearModel;
use prepare_dataset::load_dataset_from_folder;
use pmc::PMC;
use ndarray::{Array1, Array2};

fn main() {
    // === Chargement des données ===
    let dataset = load_dataset_from_folder("../dataset/train");

    let data: Vec<(Array1<f64>, f64)> = dataset
        .into_iter()
        .map(|dp| {
            let label = if dp.label == 0 { 1.0 } else { 0.0 }; // A = 1, autres = 0
            (Array1::from(dp.features), label)
        })
        .collect();

    let n_samples = data.len();
    let n_features = data[0].0.len();

    // === Modèle Linéaire ===
    let mut linear_model = LinearModel::new(n_features);
    linear_model.train(&data, 0.0001, 10);

    let test_x = &data[0].0;
    let prediction = linear_model.predict(test_x);
    let predicted_class = if prediction > 0.0 { "A" } else { "autre" };

    println!("\n=== Modèle Linéaire ===");
    println!("Résultat brut : y = {:.5}", prediction);
    println!("Classe prédite : {}", predicted_class);
    evaluate_linear(&linear_model, &data);

    // === Modèle PMC ===
    let x_array = Array2::from_shape_fn((n_samples, n_features), |(i, j)| data[i].0[j]);
    let y_array = Array1::from_shape_fn(n_samples, |i| data[i].1);

    let mut pmc_model = PMC::new(n_features, 64); 
    pmc_model.train(&x_array, &y_array, 0.001, 50); 

    let pmc_pred = pmc_model.predict(&data[0].0);
    let pmc_class = if pmc_pred > 0.5 { "A" } else { "autre" };

    println!("\n=== PMC ===");
    println!("Résultat PMC : y = {:.5}", pmc_pred);
    println!("Classe prédite : {}", pmc_class);
    evaluate_pmc(&pmc_model, &x_array, &y_array);
}

// Évaluation pour le modèle linéaire
fn evaluate_linear(model: &LinearModel, data: &[(Array1<f64>, f64)]) {
    let mut correct = 0;
    for (x, y_true) in data {
        let y_pred = model.predict(x);
        let predicted = if y_pred > 0.0 { 1.0 } else { 0.0 };
        if (predicted - y_true).abs() < 1e-6 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / data.len() as f64 * 100.0;
    println!("Précision (modèle linéaire) : {:.2}%", accuracy);
}

// Évaluation pour le PMC
fn evaluate_pmc(model: &PMC, x: &Array2<f64>, y: &Array1<f64>) {
    let mut correct = 0;
    for i in 0..x.shape()[0] {
        let pred = model.predict(&x.row(i).to_owned());
        let pred_label = if pred > 0.5 { 1.0 } else { 0.0 };
        if (pred_label - y[i]).abs() < 1e-6 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / x.shape()[0] as f64 * 100.0;
    println!("Précision (PMC) : {:.2}%", accuracy);
}
