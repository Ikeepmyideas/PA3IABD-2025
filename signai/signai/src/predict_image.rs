use image::{io::Reader as ImageReader, imageops::FilterType};
use crate::linear::LogisticRegression;

pub fn predict_dataset(models: &Vec<LogisticRegression>, input: &Vec<f64>) -> usize {
    let mut best_class = 0;
    let mut best_score = f64::MIN;

    for (i, model) in models.iter().enumerate() {
        let score = model.predict_single(input);
        let proba = 1.0 / (1.0 + (-score).exp()); // sigmoid
        if proba > best_score {
            best_score = proba;
            best_class = i;
        }
    }

    best_class
}

pub fn predict_image(image_path: &str, models: &Vec<LogisticRegression>, label_map: &Vec<String>) -> String {
    // 1. Charger et redimensionner l'image
    let img = ImageReader::open(image_path).unwrap().decode().unwrap().to_rgb8();
    let resized = image::imageops::resize(&img, 64, 64, FilterType::Nearest);

    // 2. Aplatir et normaliser les pixels
    let input: Vec<f64> = resized
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f64 / 255.0))
        .collect();

    // 3. Tester chaque modèle et trouver la meilleure classe
    let mut best_class = 0;
    let mut best_score = f64::MIN;

    for (i, model) in models.iter().enumerate() {
        let score = model.predict_single(&input);
        let proba = 1.0 / (1.0 + (-score).exp()); // sigmoid
        if proba > best_score {
            best_score = proba;
            best_class = i;
        }
    }

    // 4. Retourner le label associé à la meilleure classe
    label_map[best_class].clone()
}
