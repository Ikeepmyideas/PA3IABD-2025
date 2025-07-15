mod linear_model;
mod mlp_model;
mod rbfn_model;
mod svm_model;

use linear_model::{LinearModel, tanh, tanh_derivative};
use linear_model::{sigmoid, sigmoid_derivative};
use std::ffi::c_void;
use mlp_model::MLP;
use rbfn_model::{RBFN, RBFMode};
use svm_model::SVMClassifierRBF;
use nalgebra::{DMatrix, DVector};
use crate::svm_model::SVMMultiClassRBF;



#[no_mangle]
pub extern "C" fn create_linear_model(n_features: usize, lr: f64, epochs: usize) -> *mut c_void {
    let model = Box::new(LinearModel::new(n_features, lr, epochs));
    Box::into_raw(model) as *mut c_void
}
// === Modèle Linéaire ===
//  Régression 
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

//  Classification binaire (activation tanh) 
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


// === Modèle MLP ===

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
