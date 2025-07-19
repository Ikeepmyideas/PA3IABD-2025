mod linear_model;
mod mlp_model;
mod rbfn_model;
mod svm_model;
mod loss;

use linear_model::MultiClassLinear;
use linear_model::{LinearModel, ActivationFn};
use std::slice;
use nalgebra::{DMatrix, DVector};
use mlp_model::MLP;
use mlp_model::MLPClassifier;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::fs::File;
use std::io::{Write, Read};
use std::ptr;
use rbfn_model::{RBFN, RBFMode};
use svm_model::SVMClassifierRBF;
use crate::svm_model::SVMMultiClassRBF;

#[no_mangle]
pub extern "C" fn create_linear(
    n_features: usize,
    learning_rate: f32,
    max_epochs: usize,
    activation_type: i32,  // 1 = Tanh, 2 = Sign, autre = Linear

) -> *mut c_void {
    let activation = match activation_type {
        1 => ActivationFn::Tanh,
        2 => ActivationFn::Sign,
        _ => ActivationFn::Sign,  // Par défaut on prend Sign
    };


    let model = Box::new(LinearModel::new(n_features, learning_rate, max_epochs, activation));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_linear(
    model_ptr: *mut c_void,
    x_ptr: *const f32,
    y_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    show_logs: bool,
) {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { slice::from_raw_parts(y_ptr, n_samples) };

    // Reconstruire Vec<Vec<f32>>
    let x: Vec<Vec<f32>> = x_flat.chunks_exact(n_features).map(|c| c.to_vec()).collect();

    model.fit(&x, y, show_logs);
}

#[no_mangle]
pub extern "C" fn predict_linear(
    model_ptr: *const c_void,
    x_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    out_ptr: *mut f32,
) {
    let model = unsafe { &*(model_ptr as *const LinearModel) };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, n_samples) };

    let x: Vec<Vec<f32>> = x_flat.chunks_exact(n_features).map(|c| c.to_vec()).collect();

    let preds = model.predict(&x);
    for (i, p) in preds.iter().enumerate() {
        out[i] = *p;
    }
}

#[no_mangle]
pub extern "C" fn destroy_linear(model_ptr: *mut c_void) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr as *mut LinearModel)); }
    }
}
#[no_mangle]
pub extern "C" fn create_multiclass(
    n_classes: usize,
    n_features: usize,
    learning_rate: f32,
    max_epochs: usize,
    activation_type: i32,
) -> *mut c_void {
    let activation = match activation_type {
        1 => ActivationFn::Tanh,
        2 => ActivationFn::Sign,
        _ => ActivationFn::Sign,
    };

    let model = Box::new(MultiClassLinear::new(n_classes, n_features, learning_rate, max_epochs, activation));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_multiclass(
    model_ptr: *mut c_void,
    x_ptr: *const f32,
    y_ptr: *const u32,
    n_samples: usize,
    n_features: usize,
    show_logs: bool,
) {
    let model = unsafe { &mut *(model_ptr as *mut MultiClassLinear) };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { slice::from_raw_parts(y_ptr, n_samples) };

    let x: Vec<Vec<f32>> = x_flat.chunks_exact(n_features).map(|c| c.to_vec()).collect();
    let y_vec: Vec<usize> = y.iter().map(|&val| val as usize).collect();

    model.fit(&x, &y_vec, show_logs);
}

#[no_mangle]
pub extern "C" fn predict_multiclass(
    model_ptr: *const c_void,
    x_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    out_ptr: *mut u32,
) {
    let model = unsafe { &*(model_ptr as *const MultiClassLinear) };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, n_samples) };

    let x: Vec<Vec<f32>> = x_flat.chunks_exact(n_features).map(|c| c.to_vec()).collect();
    let preds = model.predict(&x);

    for (i, p) in preds.iter().enumerate() {
        out[i] = *p as u32;
    }
}

#[no_mangle]
pub extern "C" fn destroy_multiclass(model_ptr: *mut c_void) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr as *mut MultiClassLinear)); }
    }
}
#[no_mangle]
pub extern "C" fn create_mlp_model(
    n_inputs: usize,
    n_hidden: usize,
    learning_rate: f64,
    epochs: usize,
    is_regression: bool,
    use_activation: bool,
) -> *mut c_void {
    let model = Box::new(MLP::new(n_inputs, n_hidden, learning_rate, epochs, is_regression, use_activation));
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
#[no_mangle]
pub extern "C" fn create_mlp_classifier(
    n_inputs: usize,
    n_hidden: usize,
    n_classes: usize,
    learning_rate: f64,
    epochs: usize,
) -> *mut c_void {
    let model = Box::new(MLPClassifier::new(n_inputs, n_hidden, n_classes, learning_rate, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_mlp_classifier(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const u32,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut MLPClassifier) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec: Vec<usize> = y.iter().map(|&v| v as usize).collect();

    model.fit(&x_rows, &y_vec);
}

#[no_mangle]
pub extern "C" fn predict_mlp_classifier(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    n_features: usize,
) -> u32 {
    let model = unsafe { &*(model_ptr as *mut MLPClassifier) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec()) as u32
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
#[no_mangle]
pub extern "C" fn evaluate_mlp_classifier_mse(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const u32,
    n_samples: usize,
    n_features: usize,
) -> f32 {
    let model = unsafe { &*(model_ptr as *mut MLPClassifier) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let x_rows: Vec<Vec<f64>> = x.chunks(n_features).map(|c| c.to_vec()).collect();
    let y_vec: Vec<usize> = y.iter().map(|&val| val as usize).collect();

    match model.evaluate_mse(&x_rows, &y_vec) {
        Some(score) => score,
        None => -1.0,
    }
}
#[no_mangle]
pub extern "C" fn save_mlp_classifier(model_ptr: *mut c_void, path: *const c_char) {
    if model_ptr.is_null() || path.is_null() {
        return;
    }

    let model = unsafe { &*(model_ptr as *mut MLPClassifier) };
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(p) => p,
        Err(_) => return,
    };

    let mut file = match File::create(path_str) {
        Ok(f) => f,
        Err(_) => return,
    };

    let _ = bincode::serialize_into(&mut file, model);
}

#[no_mangle]
pub extern "C" fn load_mlp_classifier(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    let mut file = match File::open(path_str) {
        Ok(f) => f,
        Err(_) => return ptr::null_mut(),
    };

    let model: MLPClassifier = match bincode::deserialize_from(&mut file) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    let boxed = Box::new(model);
    Box::into_raw(boxed) as *mut c_void
}
#[no_mangle]
pub extern "C" fn get_mlp_loss_pointer(ptr: *const MLPClassifier) -> *const f32 {
    unsafe {
        if let Some(model) = ptr.as_ref() {
            model.loss_per_epoch.as_ptr()
        } else {
            std::ptr::null()
        }
    }
}
#[no_mangle]
pub extern "C" fn get_rbfn_loss_pointer(ptr: *const RBFN) -> *const f32 {
    unsafe {
        if let Some(model) = ptr.as_ref() {
            model.loss_per_epoch.as_ptr()
        } else {
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn get_rbfn_loss_length(ptr: *const RBFN) -> usize {
    unsafe {
        if let Some(model) = ptr.as_ref() {
            model.loss_per_epoch.len()
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn get_mlp_loss_length(ptr: *const MLPClassifier) -> usize {
    unsafe {
        if let Some(model) = ptr.as_ref() {
            model.loss_per_epoch.len()
        } else {
            0
        }
    }
}

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
#[no_mangle]
pub extern "C" fn get_svm_loss_pointer(ptr: *const SVMMultiClassRBF) -> *const f64 {
    if ptr.is_null() {
        return std::ptr::null();
    }
    let model = unsafe { &*ptr };
    let loss = model.get_average_loss_per_epoch();
    Box::into_raw(loss.into_boxed_slice()) as *const f64
}

#[no_mangle]
pub extern "C" fn get_svm_loss_length(ptr: *const SVMMultiClassRBF) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let model = unsafe { &*ptr };
    model.get_average_loss_per_epoch().len()
}
