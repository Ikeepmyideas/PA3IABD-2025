
mod linear_model;
mod mlp_model;
mod loss;
mod rbfn_model;
mod svm_model;

use linear_model::linear_gradient_descent::LinearModel;
use linear_model::linear_least_squares::LeastSquaresModel;
use linear_model::linear_pseudo_inverse::PseudoInverseModel;
use crate::linear_model::utils::activations::{sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative};
use linear_model::linear_rosenblat_method::PerceptronModel;
use crate::linear_model::multiclasses::softmax_with_gradient_descent::SoftmaxModel;
use mlp_model::MLP;
use mlp_model::{MLPClassifier, Activation};
use rbfn_model::{RBFN, RBFMode};
use svm_model::SVMClassifierRBF;
use nalgebra::{DMatrix, DVector};
use crate::svm_model::SVMMultiClassRBF;


use std::ffi::c_void;
use std::panic;
// ================================================== MODELE LINEAIRE =======================================================
/// 
/// 
/// ======== DESCENTE DE GRADIENT ========
/// 
/// ========================================= CAS DE LA REGRESSION ====================================================///
/// 
/// 
/// 
#[no_mangle]
pub extern "C" fn create_linear_model_gradient_descent(n_features: usize, lr: f64, epochs: usize) -> *mut c_void {
    let model = Box::new(LinearModel::new(n_features, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_linear_model_gradient_descent(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();
    model.fit(&x_rows, &y_vec, None, None);
}

#[no_mangle]
pub extern "C" fn predict_linear_model_gradient_descent(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec(), None)
}

/// ========================================= CAS DE LA classification binaire ====================================================///
/// 
#[no_mangle]
pub extern "C" fn train_linear_model_gradient_descent_with_activation(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    activation_id: u32,  // 0 = sigmoid, 1 = tanh, 2 = relu
) {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    let (activation, derivative): (fn(f64) -> f64, fn(f64) -> f64) = match activation_id {
        0 => (sigmoid, sigmoid_derivative),
        1 => (tanh, tanh_derivative),
        2 => (relu, relu_derivative),
        _ => (sigmoid, sigmoid_derivative), // par défaut
    };

    model.fit(&x_rows, &y_vec, Some(activation), Some(derivative));
}

#[no_mangle]
pub extern "C" fn predict_linear_model_gradient_descent_with_activation(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
    activation_id: u32,
) -> f64 {
    let model = unsafe { &*(model_ptr as *const LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };

    let activation: fn(f64) -> f64 = match activation_id {
        0 => sigmoid,
        1 => tanh,
        2 => relu,
        _ => sigmoid,
    };

    model.predict(&x.to_vec(), Some(activation))
}


/// ======== MOINDRES CARRES (analytique) ========
/// 
/// ========================================= CAS DE LA REGRESSION ====================================================///
///La moindres carrés (Least Squares) est une méthode analytique, non itérative.
///Autrement dit : dès que on appelle fit, on obtiens immédiatement les poids optimaux 
///sans besoin d'une phase d'entraînement séparée.
/// 
/// 
/// 
#[no_mangle]
pub extern "C" fn create_linear_model_least_squares(
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) -> *mut c_void {
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    let model = LeastSquaresModel::fit(&x_rows, &y_vec);
    Box::into_raw(Box::new(model)) as *mut c_void
}

#[no_mangle]
pub extern "C" fn predict_linear_model_least_squares(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *const LeastSquaresModel) }; // pas besoin de mut
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict_raw(&x.to_vec())
}

//// ========================================= CAS DE LA classification binaire ====================================================///
#[no_mangle]
pub extern "C" fn predict_linear_model_least_squares_with_activation(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
    activation_id: u32,
) -> f64 {
    let model = unsafe { &*(model_ptr as *const LeastSquaresModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };

    let activation: fn(f64) -> f64 = match activation_id {
        0 => sigmoid,
        1 => tanh,
        2 => relu,
        _ => sigmoid,
    };

    let z = model.predict_raw(&x.to_vec());
    activation(z)
}


//
// ========== PSEUDO-INVERSE ==========
// La pseudo-inverse est directement calculée via fit() 
/// ========================================= CAS DE LA REGRERSSION ====================================================///
#[no_mangle]
pub extern "C" fn create_linear_model_pseudo_inverse(
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) -> *mut c_void {
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    match PseudoInverseModel::fit(&x_rows, &y_vec) {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}


#[no_mangle]
    pub extern "C" fn predict_linear_model_pseudo_inverse(
        model_ptr: *mut c_void,
        x_ptr: *const f64,
        n_features: usize,
    ) -> f64 {
        let model = unsafe { &*(model_ptr as *const PseudoInverseModel) };
        let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };

        match model.predict_raw(x) {
            Ok(pred) => pred,
            Err(e) => {
                eprintln!("Erreur lors de la prédiction : {:?}", e);
                std::f64::NAN
            }
        }
    }
///========================================= CAS DE LA classification binaire ====================================================///
#[no_mangle]
pub extern "C" fn predict_linear_model_pseudo_inverse_with_activation(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
    activation_id: u32,
) -> f64 {
    let model = unsafe { &*(model_ptr as *const PseudoInverseModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };

    let activation: fn(f64) -> f64 = match activation_id {
        0 => sigmoid,
        1 => tanh,
        2 => relu,
        _ => sigmoid,
    };

    match model.predict_raw(&x.to_vec()) {
        Ok(z) => activation(z),
        Err(_) => std::f64::NAN,
    }
}

///
// ================== METHODE DE ROSENBLAT POUR LA CLASSIFICATION BINAIRE ============================
///
/// 
/// 
/// 
/// Créer un perceptron
#[no_mangle]
pub extern "C" fn create_perceptron_model(n_features: usize, lr: f64, epochs: usize) -> *mut c_void {
    let model = Box::new(PerceptronModel::new(n_features, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

/// Entraîner le perceptron
#[no_mangle]
pub extern "C" fn train_perceptron_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut PerceptronModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    model.fit(&x_rows, y);
}

/// Prédiction avec le perceptron (sortie binaire -1 ou +1)
#[no_mangle]
pub extern "C" fn predict_perceptron_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *const PerceptronModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(x)
}

///
// ================== CAS MULTICLASSES ============================
///
///========================== SOFTMAX + DESCENTE DE GRADIENT =======================
/// 
/// la softmax peut etre utilisé uniquement avec la descente de gradient dans notre cas car elle est non linéaire

/// Création du modèle Softmax
#[no_mangle]
pub extern "C" fn create_softmax_model(
    n_features: usize,
    n_classes: usize,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,  // régularisation L2
) -> *mut SoftmaxModel {
    let model = SoftmaxModel::new(n_features, n_classes, learning_rate, epochs, lambda);
    Box::into_raw(Box::new(model))
}

/// Entraînement du modèle avec protection contre les panics
#[no_mangle]
pub extern "C" fn train_softmax_model(
    model_ptr: *mut SoftmaxModel,
    x_ptr: *const f64,
    y_ptr: *const usize,
    n_samples: usize,
    n_features: usize,
) {
    let result = panic::catch_unwind(|| {
        assert!(!model_ptr.is_null(), "model_ptr is null");
        assert!(!x_ptr.is_null(), "x_ptr is null");
        assert!(!y_ptr.is_null(), "y_ptr is null");

        let model = unsafe { model_ptr.as_mut().unwrap() };

        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
        let mut X = vec![vec![0.0; n_features]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_features {
                X[i][j] = x_slice[i * n_features + j];
            }
        }

        let y_slice = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
        let y_vec = y_slice.to_vec();

        println!(" [Rust] Lancement de l'entraînement sur {} exemples", n_samples);
        model.fit(&X, &y_vec);
    });

    if result.is_err() {
        eprintln!(" [Rust] Panic détecté dans train_softmax_model !");
    }
}

/// Prédiction avec le modèle
#[no_mangle]
pub extern "C" fn predict_softmax_model(
    model_ptr: *const SoftmaxModel,
    x_ptr: *const f64,
    n_features: usize,
) -> usize {
    assert!(!model_ptr.is_null(), "model_ptr is null");
    assert!(!x_ptr.is_null(), "x_ptr is null");

    let model = unsafe { model_ptr.as_ref().unwrap() };
    let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    let x_vec = x_slice.to_vec();

    model.predict(&x_vec)
}


// === Modèle MLP ===

#[repr(C)]
pub enum ActivationId {
    ReLU = 0,
    Tanh = 1,
}

#[no_mangle]
pub extern "C" fn create_mlp_classifier(
    n_inputs: usize,
    n_hidden: usize,
    n_classes: usize,
    learning_rate: f64,
    epochs: usize,
    activation_id: u32, // 0 = ReLU, 1 = Tanh
) -> *mut c_void {
    

    let activation = match activation_id {
        0 => Activation::ReLU,
        1 => Activation::Tanh,
        _ => Activation::ReLU, // fallback par défaut
    };

    let model = Box::new(MLPClassifier::new(
        n_inputs,
        n_hidden,
        n_classes,
        learning_rate,
        epochs,
        activation,
    ));

    Box::into_raw(model) as *mut c_void
}




// === RBFN Model ===
// regression
#[no_mangle]
pub extern "C" fn create_rbfn_regression_model(
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(sigma, learning_rate, epochs, RBFMode::Regression));
    Box::into_raw(model) as *mut c_void
}

//classification binaire
#[no_mangle]
pub extern "C" fn create_rbfn_binary_classification_model(
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(sigma, learning_rate, epochs, RBFMode::BinaryClassification));
    Box::into_raw(model) as *mut c_void
}

//classification multiclasses
#[no_mangle]
pub extern "C" fn create_rbfn_multiclass_model(
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
    n_classes: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(sigma, learning_rate, epochs, RBFMode::MultiClassification(n_classes)));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_rbfn_model_auto(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    n_outputs: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut RBFN) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples * n_outputs) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();

    match &model.mode {
        RBFMode::Regression | RBFMode::BinaryClassification => {
            let y_vec: Vec<f64> = y.iter().copied().collect();
            model.fit_closed_form(&x_rows, &y_vec);
        }
        RBFMode::MultiClassification(_) => {
            let y_rows: Vec<Vec<f64>> = y.chunks(n_outputs).map(|c| c.to_vec()).collect();
            model.fit_gradient_descent(&x_rows, &y_rows);
        }
    }
}

#[no_mangle]
pub extern "C" fn predict_rbfn_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut RBFN) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict_label(&x.to_vec())
}



// === SVM RBF ===


#[no_mangle]
pub extern "C" fn create_svm_rbf_classifier(
    gamma: f64,
    c: f64,
    lr: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(SVMClassifierRBF::new(gamma, c, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_svm_rbf_classifier(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut SVMClassifierRBF) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_vec: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    model.fit(&x_vec, &y_vec);
}

#[no_mangle]
pub extern "C" fn predict_svm_rbf_classifier(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut SVMClassifierRBF) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec())
}



// SVM multiclasses
#[no_mangle]
pub extern "C" fn create_svm_rbf_multiclass(
    gamma: f64,
    c: f64,
    lr: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(SVMMultiClassRBF::new(gamma, c, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn destroy_svm_rbf_multiclass(model_ptr: *mut c_void) {
    if !model_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(model_ptr as *mut SVMMultiClassRBF);
        }
    }
}

#[no_mangle]
pub extern "C" fn train_svm_rbf_multiclass(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const usize,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut SVMMultiClassRBF) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_vec = x.chunks(n_features).map(|c| c.to_vec()).collect::<Vec<_>>();
    model.fit(&x_vec, &y.to_vec());
}

#[no_mangle]
pub extern "C" fn predict_svm_rbf_multiclass(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> usize {
    let model = unsafe { &*(model_ptr as *mut SVMMultiClassRBF) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(x)
}

#[no_mangle]
pub extern "C" fn train_and_predict_svm_rbf_multiclass(
    x_ptr: *const f64,
    y_ptr: *const usize,
    n_samples: usize,
    n_features: usize,
    gamma: f64,
    c: f64,
    lr: f64,
    epochs: usize,
    test_point_ptr: *const f64,
) -> usize {
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let test_point = unsafe { std::slice::from_raw_parts(test_point_ptr, n_features) };
    let x_vec = x.chunks(n_features).map(|c| c.to_vec()).collect::<Vec<_>>();
    let mut model = SVMMultiClassRBF::new(gamma, c, lr, epochs);
    model.fit(&x_vec, &y.to_vec());
    model.predict(test_point)
}

#[no_mangle]
pub extern "C" fn train_and_predict_svm_rbf_multiclass_grid(
    x_ptr: *const f64,
    y_ptr: *const usize,
    n_samples: usize,
    n_features: usize,
    gamma: f64,
    c: f64,
    lr: f64,
    epochs: usize,
    grid_ptr: *const f64,
    n_grid_points: usize,
) -> *mut usize {
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let grid = unsafe { std::slice::from_raw_parts(grid_ptr, n_grid_points * n_features) };

    let x_vec = x.chunks(n_features).map(|c| c.to_vec()).collect::<Vec<_>>();
    let grid_vec = grid.chunks(n_features).map(|c| c.to_vec());

    let mut model = SVMMultiClassRBF::new(gamma, c, lr, epochs);
    model.fit(&x_vec, &y.to_vec());

    let preds = grid_vec.map(|pt| model.predict(&pt)).collect::<Vec<_>>();
    let boxed = preds.into_boxed_slice();
    Box::into_raw(boxed) as *mut usize
}

#[no_mangle]
pub extern "C" fn destroy_usize_array(ptr: *mut usize, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}



//
// ========== FONCTIONS AFFICHAGE POIDS ET BIAS ==========
// 


#[no_mangle]
pub extern "C" fn get_model_bias(model_ptr: *mut c_void) -> f64 {
    let model = unsafe { &*(model_ptr as *const PseudoInverseModel) };
    model.bias
}

#[no_mangle]
pub extern "C" fn get_model_weights_ptr(model_ptr: *mut c_void) -> *const f64 {
    let model = unsafe { &*(model_ptr as *const PseudoInverseModel) };
    model.weights.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_model_weights_len(model_ptr: *mut c_void) -> usize {
    let model = unsafe { &*(model_ptr as *const PseudoInverseModel) };
    model.weights.len()
}

