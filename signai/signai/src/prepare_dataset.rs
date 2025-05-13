use image::{GenericImageView, imageops::FilterType};
use walkdir::WalkDir;
use std::fs::File;
use csv::Writer;

#[derive(Clone)]
/// Structure représentant une donnée d entrainement ou de test
pub struct DataPoint {
    pub features: Vec<f64>, // pixels normalisés
    pub label: usize,        // entier de 0 à 35
}

/// Charge toutes les images dans un dossier 
/// Chaque sous-dossier représente une classe : "A", "B", ..., "Z", "0", ..., "9"
pub fn load_dataset_from_folder(folder_path: &str) -> Vec<DataPoint> {
    let mut dataset = Vec::new();

    for entry in WalkDir::new(folder_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path().is_file()
                && e.path().extension().map_or(false, |ext| {
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                })
        })
    {
        let path = entry.path();
        let label_str = match path.parent().and_then(|p| p.file_name()) {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        // Convertit "A" → 0, ..., "Z" → 25, "0" → 26, ..., "9" → 35
        let c = label_str.chars().next().unwrap();

        let label = match c {
            'A'..='Z' => (c as u8 - b'A') as usize,
            '0'..='9' => (c as u8 - b'0') as usize + 26,
            _ => continue, // caractère inconnu
        };

        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => continue,
        };

        let resized = image::imageops::resize(&img, 64, 64, FilterType::Nearest);

        let features: Vec<f64> = resized
            .pixels()
            .flat_map(|p| p.0.iter().map(|v| *v as f64 / 255.0))
            .collect();

        dataset.push(DataPoint { features, label });
    }

    dataset
}

/// genere un fichier CSV à partir du dossier `dataset`, pour usage avec train_one_vs_all
pub fn generate_dataset() {
    let mut wtr = Writer::from_path("dataset_rgb.csv").expect("Erreur de création CSV");

    let headers: Vec<String> = (0..(64 * 64 * 3))
        .map(|i| format!("p{}", i))
        .chain(std::iter::once("label".into()))
        .collect();
    wtr.write_record(&headers).unwrap();

    for entry in WalkDir::new("dataset")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file() &&
            e.path().extension().map_or(false, |ext| {
                matches!(ext.to_str().unwrap_or("").to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp")
            })
    })
    {
        let path = entry.path();
        let label = match path.parent().and_then(|p| p.file_name()) {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("Erreur lecture image {}: {:?}", path.display(), e);
                continue;
            }
        };

        let resized = image::imageops::resize(&img, 64, 64, FilterType::Nearest);

        let flat_pixels: Vec<String> = resized
            .pixels()
            .flat_map(|p| p.0.iter().map(|v| format!("{:.5}", *v as f64 / 255.0)))
            .collect();

        let mut row = flat_pixels;
        row.push(label);
        wtr.write_record(&row).unwrap();
    }

    wtr.flush().unwrap();
    println!("dataset_rgb.csv généré avec succès.");
}
