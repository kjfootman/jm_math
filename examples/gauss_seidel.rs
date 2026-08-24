#![allow(non_snake_case)]
use jm_lib::prelude::*;

fn main() -> Result<(), Error> {
    env_logger::builder().is_test(false).try_init()?;

    let M = CSRMatrix::from_mtx("resources/mtx/bcsstk16.mtx")?;
    let b = get_source(&M);
    let gs = GaussSeidelBuilder::new()
        .iter_max(5000)
        .tolerance(1E-12)
        .build();

    match gs.solve(&M, &b) {
        Ok(_) => {
            log::info!(
                "Converged - iteration: {}, residual: {:.2E}",
                gs.iter(),
                gs.residual()
            );
        }
        Err(e) => {
            log::error!("{e}");
        }
    }

    Ok(())
}

fn get_source(M: &CSRMatrix) -> Vector {
    let ia = M.row_ptr();
    let aa = M.values();

    let arr = ia
        .windows(2)
        .map(|range| {
            let start = range[0];
            let end = range[1];

            aa[start..end].iter().sum::<f64>()
        })
        .collect::<Vec<_>>();

    Vector::from(arr)
}
