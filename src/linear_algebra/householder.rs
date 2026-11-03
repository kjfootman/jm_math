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

pub fn transform(u: &[f64], v: &mut [f64]) {
    let arch = simd::arch();
    let sigma = arch.dispatch(simd::VectorDot(u, v));

    // arch.dispatch(simd::VectorScaleAdd(inplace, v, -2.0 * sigma, u));
    arch.dispatch(simd::VectorScaleAddAssign(v, -2.0 * sigma, u));
}

#[cfg(test)]
mod tests {
    // use crate::prelude::{Vector};
    use super::*;

    #[test]
    fn househoder_transform() {
        // TODO: rust_decimal 크레이트를 사용해서 부동소수점 오차 해결
        let u = vec![0.859457, 0.138848, 0.323978, 0.370260];
        let mut v = vec![6.0, 3.0, 7.0, 8.0];
        let mag = u.iter().map(|&value| value * value).sum::<f64>().sqrt();

        // assert |u| = 1.0
        assert_eq!((1000.0 * mag).round() / 1000.0, 1.0);

        transform(&u, &mut v);

        // 소수점 5째 자리에서 반올림
        v.iter_mut()
            .for_each(|v| *v = (10000.0 * *v).round() / 10000.0);

        assert_eq!(v, vec![-12.5698, 0.0, 0.0, 0.0]);
    }
}
