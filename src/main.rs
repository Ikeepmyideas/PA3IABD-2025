mod linear_model;
mod mlp_model;
mod svm_model;
mod prepare_dataset;

use linear_model::{LinearModel, tanh, tanh_derivative};
use mlp_model::MLP;
use svm_model::{SVMClassifierRBF, SVMMultiClassRBF};


fn main() {

    // === 1. Génération du dataset CSV à partir du dossier d’images
    prepare_dataset::generate_dataset_train();
    prepare_dataset::generate_dataset_test();

    // Exemple de données
    let X = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![1.5, 1.5]]; //x une matrice 3*2 (nbr de features = 2) et on a trois exemples / échantilloons
    let y_regression = vec![3.0, 3.0, 3.0]; //y_regression = un vecteur 3*1
    let y_classification = vec![1.0, -1.0, 1.0]; //y_classification = un vecteur 3*1 (il prend les valuers 1 et -1)

    
    // --- Linear Model pour regression
    let mut lm = LinearModel::new(2, 0.01, 1000);
    lm.fit(&X, &y_regression, None, None); // pas d'activation
    println!("Linear regression prediction: {}", lm.predict(&X[0], None)); //on teste sur le premier exemple : predict [1.0, 2.0] -> 3.0
    
    // --- Linear Model pour classification (avec tanh)
    let mut lm_classif = LinearModel::new(2, 0.9, 1000);
    lm_classif.fit(&X, &y_classification, Some(tanh), Some(tanh_derivative));
    println!("Linear classification prediction: {}", lm_classif.predict(&X[1], Some(tanh)));
    
    // === Cas de test : fonction XOR
    let X = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];
    let y = vec![-1.0, 1.0, 1.0, -1.0];  // Sorties XOR (codées en -1 / +1)

    // === Test MLP sur XOR
    let mut mlp = MLP::new(2, 4, 0.1, 5000);  // 2 entrées, 4 neurones cachés, lr = 0.1, 5000 epochs
    mlp.fit(&X, &y);

    println!("=== Prédictions MLP sur XOR ===");
    for (i, x) in X.iter().enumerate() {
        let pred = mlp.predict(x);
        let label = if pred >= 0.0 { 1.0 } else { -1.0 };
        println!("Exemple {} : Entrée = {:?}, Prédit = {:.3}, Classe = {}, Vrai = {}", 
            i, x, pred, label, y[i]);
    }
   
    
    // === Test SVM binaire avec noyau RBF ===
    println!("\n=== Test SVM Binaire RBF ===");
    let x_svm = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_svm = vec![-1.0, 1.0, 1.0, -1.0]; //XOR commme pour le mlp

    let mut svm = SVMClassifierRBF::new(0.5, 1.0, 0.1, 100);
    svm.fit(&x_svm, &y_svm);

    for(i, x)in x_svm.iter().enumerate(){
        let pred = svm.predict(x);
        println!("Test {} : Entrée {:?}, Prédit = {}, Vrai = {}", i,x,pred,y_svm[i]);
    }


    // === Test SVM Multiclasse RBF ===
    println!("\n=== Test SVM Multiclasse RBF ===");
    let x_multi = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![8.0, 8.0],
        vec![9.0, 9.0],
        vec![5.0, 1.0],
        vec![6.0, 2.0],
    ];
    let y_multi = vec![0,0,1,1,2,2]; // On prend 3 classe 0,1 et 2

    let mut svm_multi = SVMMultiClassRBF::new(0.5,1.0,0.05,200);
    svm_multi.fit(&x_multi,&y_multi);
    
    for(i,x) in x_multi.iter().enumerate(){
        let pred = svm_multi.predict(x);
        println!("Test {} : Entrée = {:?}, Prédit = {}, Vrai = {}",i,x,pred,y_multi[i]);
    }
}
