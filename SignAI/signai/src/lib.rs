mod linear_model;
mod mlp_model;
mod rbfn_model;
mod svm_model;

use std::ffi::c_void;
use std::ffi::{c_char, CStr};
use nalgebra::{DMatrix, DVector};

use linear_model::{evaluate_accuracy as evaluate_accuracy_linear, tanh,tanh_derivative, sigmoid, LinearModel};
use mlp_model::{evaluate_accuracy_mlp, MLP};
use rbfn_model::{evaluate_accuracy_rbfn, RBFN,RBFMode};
use svm_model::{evaluate_accuracy_svm, SVMClassifierRBF};
use linear_model::{sigmoid_derivative};

#[no_mangle]
pub extern "C" fn create_linear_model(n_features: usize, lr: f64, epochs: usize) -> *mut c_void {
    let model = Box::new(LinearModel::new(n_features, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

//  Régression linéaire 
#[no_mangle]
pub extern "C" fn train_linear_model(
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

    model.fit(&x_rows, &y_vec, None, None); // pas d'activation → régression
}

#[no_mangle]
pub extern "C" fn predict_linear_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec(), None)
}

// Classification binaire (activation tanh) 
#[no_mangle]
pub extern "C" fn train_linear_model_classification(
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

    model.fit(&x_rows, &y_vec, Some(tanh), Some(tanh_derivative)); // avec activation tanh
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec(), Some(tanh))
}


//  Classification multiclasses (activation sigmoide) 
#[no_mangle]
pub extern "C" fn train_linear_model_sigmoid(
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

    model.fit(&x_rows, &y_vec, Some(sigmoid), Some(sigmoid_derivative));
}

#[no_mangle]
pub extern "C" fn predict_linear_model_sigmoid(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec(), Some(sigmoid))
}


// Modele MLP 

#[no_mangle]
pub extern "C" fn create_mlp_model(
    n_inputs: usize,
    n_hidden: usize,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(MLP::new(n_inputs, n_hidden, learning_rate, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_mlp_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut MLP) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    model.fit(&x_rows, &y_vec);
}

#[no_mangle]
pub extern "C" fn predict_mlp_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut MLP) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec())
}


// RBFN Model 
#[no_mangle]
pub extern "C" fn create_rbfn_regression_model(
    n_hidden: usize,
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(n_hidden, sigma, learning_rate, epochs, RBFMode::Regression));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn create_rbfn_binary_classification_model(
    n_hidden: usize,
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(n_hidden, sigma, learning_rate, epochs, RBFMode::BinaryClassification));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn create_rbfn_multiclass_model(
    n_hidden: usize,
    sigma: f64,
    learning_rate: f64,
    epochs: usize,
    n_classes: usize,
) -> *mut c_void {
    let model = Box::new(RBFN::new(n_hidden, sigma, learning_rate, epochs, RBFMode::MultiClassification(n_classes)));
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


#[no_mangle]
pub extern "C" fn create_svm_rbf_classifier(
    n_support: usize,
    gamma: f64,
    c: f64,
    lr: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(SVMClassifierRBF::new(n_support, gamma, c, lr, epochs));
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

#[no_mangle]
pub extern "C" fn save_linear_model(model: *mut LinearModel, path: *const c_char) {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model_ref = unsafe { &*model };
    model_ref.save(path_str).unwrap();
}

#[no_mangle]
pub extern "C" fn load_linear_model(path: *const c_char) -> *mut LinearModel {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model = LinearModel::load(path_str).unwrap();
    Box::into_raw(Box::new(model))
}
#[no_mangle]
pub extern "C" fn save_mlp_model(model: *mut MLP, path: *const c_char) {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model_ref = unsafe { &*model };
    model_ref.save(path_str).unwrap();
}

#[no_mangle]
pub extern "C" fn load_mlp_model(path: *const c_char) -> *mut MLP {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model = MLP::load(path_str).unwrap();
    Box::into_raw(Box::new(model))
}
#[no_mangle]
pub extern "C" fn save_rbfn_model(model: *mut RBFN, path: *const c_char) {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model_ref = unsafe { &*model };
    model_ref.save(path_str).unwrap();
}

#[no_mangle]
pub extern "C" fn load_rbfn_model(path: *const c_char) -> *mut RBFN {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model = RBFN::load(path_str).unwrap();
    Box::into_raw(Box::new(model))
}
#[no_mangle]
pub extern "C" fn save_svm_model(model: *mut SVMClassifierRBF, path: *const c_char) {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model_ref = unsafe { &*model };
    model_ref.save(path_str).unwrap();
}

#[no_mangle]
pub extern "C" fn load_svm_model(path: *const c_char) -> *mut SVMClassifierRBF {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    let model = SVMClassifierRBF::load(path_str).unwrap();
    Box::into_raw(Box::new(model))
}
#[no_mangle]
pub extern "C" fn evaluate_linear_model_accuracy(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    variant: *const c_char, 
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    let variant_str = unsafe { CStr::from_ptr(variant).to_str().unwrap() };
    let activation: Option<fn(f64) -> f64> = match variant_str {
        "tanh" => Some(tanh as fn(f64) -> f64),
        "sigmoid" => Some(sigmoid as fn(f64) -> f64),
        _ => None,
    };


    evaluate_accuracy_linear(model, &x_rows, &y_vec, activation)
}
#[no_mangle]
pub extern "C" fn evaluate_mlp_model_accuracy(
    model_ptr: *mut MLP,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut MLP) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    evaluate_accuracy_mlp(model, &x_rows, &y_vec)
}
#[no_mangle]
pub extern "C" fn evaluate_rbfn_model_accuracy(
    model_ptr: *mut RBFN,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut RBFN) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    evaluate_accuracy_rbfn(model, &x_rows, &y_vec)
}
#[no_mangle]
pub extern "C" fn evaluate_svm_model_accuracy(
    model_ptr: *mut SVMClassifierRBF,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    let model = unsafe { &*(model_ptr as *mut SVMClassifierRBF) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };
    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec = y.to_vec();

    evaluate_accuracy_svm(model, &x_rows, &y_vec)
}