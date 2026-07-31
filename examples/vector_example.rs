use jm_lib::linear_algebra::Vector;

fn main() {
    let mut v = Vector::new(5);
    v[0] = 0.0;
    v[1] = 1.0;
    v[2] = 2.0;
    v[3] = 3.0;
    v[4] = 4.0;

    println!("{v:#?}");
}
