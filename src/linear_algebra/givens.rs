pub fn compute_givens_rotation(a: f64, b: f64) -> (f64, f64) {
    let r = (a.powi(2) + b.powi(2)).sqrt();
    (a / r, b / r)
}

pub fn apply_givens_rotation(h0: f64, h1: f64, c: f64, s: f64) -> (f64, f64) {
    (h0 * c + h1 * s, -h0 * s + h1 * c)
}
