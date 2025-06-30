mod linear_model;
mod prepare_dataset;

use linear_model::{LinearModel, ActivationFn};
use csv::Reader;

fn main() {
    // Génération des CSV
    prepare_dataset::generate_dataset_train();
    prepare_dataset::generate_dataset_test();

    // Chargement des données
    let (x_train, y_train) = load_dataset("dataset_train.csv");
    let (x_test, y_test) = load_dataset("dataset_test.csv");

    if x_train.is_empty() || x_test.is_empty() {
        println!("Erreur : dataset vide.");
        return;
    }

    // Création et entraînement du modèle
    let mut model = LinearModel::new(x_train[0].len(), 0.1, 1000, ActivationFn::Sign);
    model.fit(&x_train, &y_train, true);

    // Prédiction
    let preds = model.predict(&x_test);

    // Résultats
    for (i, (p, y)) in preds.iter().zip(y_test.iter()).enumerate() {
        println!("Exemple {} → vrai: {}, prédit: {:.3}", i, y, p);
    }
}

// Lecture d'un dataset CSV
fn load_dataset(path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = Reader::from_path(path).unwrap();
    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let features: Vec<f32> = record.iter()
            .take(record.len() - 1)
            .map(|v| v.parse().unwrap_or(0.0))
            .collect();

        let label = record.get(record.len() - 1).unwrap().parse().unwrap_or(0.0);
        x.push(features);
        y.push(if label == 0.0 { 1.0 } else { -1.0 });
    }

    (x, y)
}
