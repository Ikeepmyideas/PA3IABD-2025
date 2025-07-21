mod linear_model;
mod mlp_model;
mod svm_model;
mod prepare_dataset;
mod loss;

use linear_model::{LinearModel, tanh, tanh_derivative};
use mlp_model::MLP;
use svm_model::{SVMClassifierRBF, SVMMultiClassRBF};
use loss::plot_loss_curve;

fn main() {
    // === Génération du dataset (préparation image -> CSV)
    prepare_dataset::generate_dataset_train();
    prepare_dataset::generate_dataset_test();

    // === Exemple simple : LinearModel
    let X = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![1.5, 1.5]];
    let y_regression = vec![3.0, 3.0, 3.0];
    let y_classification = vec![1.0, -1.0, 1.0];

    let mut lm = LinearModel::new(2, 0.01, 1000);
    lm.fit(&X, &y_regression, None, None);
    println!("Linear regression prediction: {}", lm.predict(&X[0], None));

    let mut lm_classif = LinearModel::new(2, 0.9, 1000);
    lm_classif.fit(&X, &y_classification, Some(tanh), Some(tanh_derivative));
    println!("Linear classification prediction: {}", lm_classif.predict(&X[1], Some(tanh)));

    // === MLP sur XOR
    let X_xor = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];
    let y_xor = vec![-1.0, 1.0, 1.0, -1.0];

    let mut mlp = MLP::new(2, 4, 0.1, 5000);
    mlp.fit(&X_xor, &y_xor);

    println!("\n=== Prédictions MLP sur XOR ===");
    for (i, x) in X_xor.iter().enumerate() {
        let pred = mlp.predict(x);
        let label = if pred >= 0.0 { 1.0 } else { -1.0 };
        println!("Ex {}: Entrée = {:?}, Prédit = {:.3}, Classe = {}, Vrai = {}", i, x, pred, label, y_xor[i]);
    }

    // === SVM Binaire RBF sur XOR
    println!("\n=== Test SVM Binaire RBF ===");
    let mut svm = SVMClassifierRBF::new(0.5, 1.0, 0.1, 100);
    svm.fit(&X_xor, &y_xor);

    for (i, x) in X_xor.iter().enumerate() {
        let pred = svm.predict(x);
        println!("Test {} : Entrée {:?}, Prédit = {}, Vrai = {}", i, x, pred, y_xor[i]);
    }

    //  Générer la courbe de loss pour le SVM
    plot_loss_curve(&svm.loss_history, "loss_plot.png");

    // === SVM Multiclasse RBF
    println!("\n=== Test SVM Multiclasse RBF ===");
    let x_multi = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![8.0, 8.0],
        vec![9.0, 9.0],
        vec![5.0, 1.0],
        vec![6.0, 2.0],
    ];
    let y_multi = vec![0, 0, 1, 1, 2, 2];

    let mut svm_multi = SVMMultiClassRBF::new(0.5, 1.0, 0.05, 200);
    svm_multi.fit(&x_multi, &y_multi);

    for (i, x) in x_multi.iter().enumerate() {
        let pred = svm_multi.predict(x);
        println!("Test {} : Entrée = {:?}, Prédit = {}, Vrai = {}", i, x, pred, y_multi[i]);
    }
}
