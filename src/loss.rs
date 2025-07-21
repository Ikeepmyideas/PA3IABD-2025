use plotters::prelude::*;

// === Visualisation ===
pub fn plot_loss_curve(loss_history: &[f32], file_path: &str) {
    let root = BitMapBackend::new(file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_loss = loss_history.iter().cloned().fold(f32::MIN, f32::max);
    let min_loss = loss_history.iter().cloned().fold(f32::MAX, f32::min);

    let mut chart = ChartBuilder::on(&root)
        .caption("Courbe de Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..loss_history.len(), min_loss..max_loss)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            loss_history.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &RED,
        ))
        .unwrap()
        .label("Loss")
        .legend(|(x, y)| Path::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw().unwrap();
}

// === Fonctions de métriques ===
pub fn mse(pred: &[f32], target: &[f32]) -> Option<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return None;
    }
    let sum: f32 = pred.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    Some(sum / pred.len() as f32)
}

pub fn accuracy(pred: &[f32], target: &[f32]) -> Option<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return None;
    }

    let mut correct = 0;
    for (p, t) in pred.iter().zip(target.iter()) {
        if p.round() == t.round() {
            correct += 1;
        }
    }

    Some(correct as f32 / pred.len() as f32)
}
