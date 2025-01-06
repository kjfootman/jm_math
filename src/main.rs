use jm_math::linear_algebra::{MSolver, Matrix, PreconType, Vector};
use std::error::Error;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut start;
    let mut elapsed;

    // set system of equations
    let A = Matrix::from_matrix_market("matrix_makert/mcfe.mtx")?;
    let (_, n) = A.dim();

    // set preconditioner
    // let P = PreconType::SOR(1.0);
    let P = PreconType::ILU0;
    let b = Vector::from_iter(vec![1.0; n]);
    let b = &A * b;
    let restart = 5;
    let ms = MSolver::build(&A, &b)
        .max_iter(500)
        .tolerance(1E-10)
        .preconditioner(P)
        .finish();

    // GMRES
    start = std::time::Instant::now();
    let x = ms.GMRES(restart)?;
    elapsed = start.elapsed().as_secs_f32();
    // println!(
    //     "GMRES({}) iter: {}, residual: {:.4E}",
    //     restart,
    //     ms.get_last_iter(),
    //     ms.get_residual()
    // );
    println!("{:.4}", x.iter().sum::<f64>());
    // println!("{x:.4}");
    println!("elapsed: {:.4} sec", elapsed);

    // HGMRES
    start = std::time::Instant::now();
    let x = ms.HGMRES(restart)?;
    elapsed = start.elapsed().as_secs_f32();
    // println!(
    //     "HGMRES({}) iter: {}, residual: {:.4E}",
    //     restart,
    //     ms.get_last_iter(),
    //     ms.get_residual()
    // );
    println!("{:.4}", x.iter().sum::<f64>());
    // println!("{x:.4}");
    println!("elapsed: {:.4} sec", elapsed);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Read, path::PathBuf};
    use rayon::prelude::*;
    const DATA_PATH: &str ="/Users/h1007185/workspace/Rust/matrix/big_size";

    #[test]
    fn big_size_matrix_test() -> Result<(), Box<dyn Error>> {
        let data_path = PathBuf::from(DATA_PATH);
        let mut file_aa = std::fs::File::open(data_path.join("aa.dat"))?;
        let mut file_bb = std::fs::File::open(data_path.join("bb.dat"))?;
        let mut file_cl = std::fs::File::open(data_path.join("cl.dat"))?;
        let mut file_rl = std::fs::File::open(data_path.join("rl.dat"))?;

        let mut buf = String::new();

        let fsize = file_aa.read_to_string(&mut buf)?;
        let aa = buf.par_split_whitespace().map(|val| val.parse::<f64>().unwrap()).collect::<Vec<_>>();
        println!("file size: {fsize} aa length: {}", aa.len());
        buf.clear();

        let fsize = file_bb.read_to_string(&mut buf)?;
        let bb = buf.par_split_whitespace().map(|val| val.parse::<f64>().unwrap()).collect::<Vec<_>>();
        println!("file size: {fsize} bb length: {}", bb.len());
        buf.clear();

        let fsize = file_cl.read_to_string(&mut buf)?;
        let cl = buf.par_split_whitespace().map(|val| val.parse::<usize>().unwrap()).collect::<Vec<_>>();
        println!("file size: {fsize} cl length: {}", cl.len());
        buf.clear();

        let fsize = file_rl.read_to_string(&mut buf)?;
        let rl = buf.par_split_whitespace().map(|val| val.parse::<usize>().unwrap()).collect::<Vec<_>>();
        println!("file size: {fsize} rl length: {}", rl.len());
        buf.clear();

        Ok(())
    }
}
