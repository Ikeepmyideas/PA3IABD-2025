mod mlp_model;
mod rbfn_model;
mod loss;
use std::ffi::c_void;
use std::slice;
use nalgebra::{DMatrix, DVector};
use mlp_model::MLP;
use mlp_model::MLPClassifier;

use rbfn_model::{RBFN, RBFMode};
fn main() {
    let x = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
    ];
    let y = vec![1.0, -1.0, -1.0];

    let mut model = RBFN::new(1.0, 0.1, 100, RBFMode::BinaryClassification);
    model.fit_closed_form(&x, &y);

    for xi in &x {
        let pred = model.predict_label(xi);
        println!("➡️ Entrée : {:?} => Prédiction : {}", xi, pred);
    }
}
