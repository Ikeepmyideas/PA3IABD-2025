use image::{GenericImageView, imageops::FilterType};
use walkdir::WalkDir;
use std::fs::File;
use csv::Writer;
use std::collections::HashMap;

/// Fonction générique utilisée par les deux versions
fn generate_dataset_from_path(dataset_path: &str, output_csv: &str) {
    let mut wtr = Writer::from_path(output_csv).expect("Erreur de création CSV");

    // Écriture de l'en-tête
    let headers: Vec<String> = (0..(64 * 64 * 3))
        .map(|i| format!("p{}", i))
        .chain(std::iter::once("label".into()))
        .collect();
    wtr.write_record(&headers).unwrap();


    // Compteur d'images par label (clé = usize pour le label)
    let mut label_counters: HashMap<usize, usize> = HashMap::new();

    for entry in WalkDir::new(dataset_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().map_or(false, |ext| {
                    matches!(ext.to_str().unwrap_or("").to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp")
                })
        })
    {
        let path = entry.path();

        // Récupération du nom du dossier parent (A, B, ..., 0, 1, ...)
        let label_str = match path.parent().and_then(|p| p.file_name()) {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        let c = match label_str.chars().next() {
            Some(ch) => ch,
            None => continue,
        };

        let label = match c {
            'A'..='Z' => (c as u8 - b'A') as usize,  // 'A' → 0, ..., 'Z' → 25
            '0'..='9' => (c as u8 - b'0') as usize + 26, // '0' → 26, ..., '9' → 35
            _ => continue,
        };

        // Vérifie si on a déjà 30 images pour ce label
        let counter = label_counters.entry(label).or_insert(0);

        // Garde toutes les images pour 'A' (label == 0), limite les autres à 30
        if label != 0 && *counter >= 30 {
            continue;
        }

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
        row.push(label.to_string());

        wtr.write_record(&row).unwrap();


        // Incrémente le compteur pour ce label
        *counter += 1;
    }

    wtr.flush().unwrap();
    println!("{} généré avec succès !", output_csv);
}

/// Génére le CSV pour l'entraînement
pub fn generate_dataset_train() {
    generate_dataset_from_path("dataset/train", "dataset_train.csv");
}

/// Génére le CSV pour les tests
pub fn generate_dataset_test() {
    generate_dataset_from_path("dataset/test", "dataset_test.csv");
}
