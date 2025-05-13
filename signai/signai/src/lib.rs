mod linear;
use linear::LinearModel;
use std::ffi::c_double;
use std::slice;
use ndarray::Array1;

mod pmc;
use pmc::PMC;

/// Exporte une prédiction utilisant LinearModel et les poids donnés
#[unsafe(no_mangle)]
pub extern "C" fn predict_linear_model(
    inputs_ptr: *const c_double,
    weights_ptr: *const c_double,
    n: usize,
    bias: c_double,
) -> c_double {
    let inputs = unsafe { slice::from_raw_parts(inputs_ptr, n) };
    let weights = unsafe { slice::from_raw_parts(weights_ptr, n) };

    let x = Array1::from(inputs.to_vec());
    let w = Array1::from(weights.to_vec());

    let model = LinearModel {
        weights: w,
        bias,
    };

    model.predict(&x)
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_pmc_model(
    inputs_ptr: *const c_double,
    n: usize,
) -> c_double {
    let inputs = unsafe { slice::from_raw_parts(inputs_ptr, n) };
    let input = Array1::from(inputs.to_vec());

    let pmc = PMC::new(n, 5);
    pmc.predict(&input)
}
