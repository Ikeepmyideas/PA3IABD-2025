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
