pub fn round(x: f64, digit: i32) -> f64 {
    let c = 10_f64.powi(digit);

    (c * x).round() / c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_test() {
        let x = 1.25346;

        assert_eq!(1_f64, round(x, 0));
        assert_eq!(1.3, round(x, 1));
        assert_eq!(1.25, round(x, 2));
    }
}
