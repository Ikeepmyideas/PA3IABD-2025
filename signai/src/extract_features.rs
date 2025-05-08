use image::GenericImageView; // obtenir les dimensions 
use image::Pixel; // manipuler les pixels 

pub fn extract_features(path: &str) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let img = image::open(path)?; // ouvrir l image, "?" -> image n existe pas 
    let (width, height) = img.dimensions(); // recupere les dimensions 
    let mut white_pixel_count = 0;
    let mut total_pixels = 0;

    for pixel in img.pixels() { // parcour chaque pixel de l'image 
        let rgb = pixel.2.to_rgb();
        let brightness = (rgb[0] as u32 + rgb[1] as u32 + rgb[2] as u32) / 3;
        if brightness > 200 {
            white_pixel_count += 1;
        }
        total_pixels += 1;
    }

    let white_ratio = white_pixel_count as f64 / total_pixels as f64;
    let aspect_ratio = width as f64 / height as f64;

    Ok((white_ratio, aspect_ratio))
}
