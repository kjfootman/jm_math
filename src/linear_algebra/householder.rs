use crate::linear_algebra::simd;

pub fn calc_householder_vec(v: &[f64], inplace: &mut [f64]) {
    debug_assert_eq!(v.len(), inplace.len());
    // debug_assert!(!v.is_empty());

    let arch = simd::arch();
    let mut magnitude = arch.dispatch(simd::VectorDot(v, v)).sqrt();

    if magnitude == 0.0 {
        return;
    }

    let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };

    inplace[0] = v[0] + sign * magnitude;
    magnitude = 2.0 * magnitude * (magnitude + v[0].abs());
    magnitude = magnitude.sqrt();

    arch.dispatch(simd::VectorScale(
        &mut inplace[1..],
        magnitude.recip(),
        &v[1..],
    ));

    inplace[0] *= magnitude.recip();
}

pub fn transform(u: &[f64], v: &[f64], inplace: &mut [f64]) {
    let arch = simd::arch();
    let sigma = arch.dispatch(simd::VectorDot(u, v));

    arch.dispatch(simd::VectorScaleAdd(inplace, v, -2.0 * sigma, u));
}
