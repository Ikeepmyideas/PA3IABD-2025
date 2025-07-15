mod linear_model;
mod mlp_model;
mod rbfn_model;

use linear_model::MultiClassLinear;
use linear_model::{LinearModel, ActivationFn};
use std::ffi::c_void;
use std::slice;
use nalgebra::{DMatrix, DVector};
use mlp_model::MLP;
use mlp_model::MLPClassifier;

use rbfn_model::{RBFN, RBFMode};

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
