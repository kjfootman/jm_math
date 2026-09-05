#![allow(non_snake_case)]
use jm_lib::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Error> {
    env_logger::builder().is_test(false).try_init()?;

    let M = CSRMatrix::from_mtx("resources/mtx/bcsstk16.mtx")?;
    let b = get_source(&M);
    let mut gs = GaussSeidelBuilder::new()
        .with_max_iter(5000)
        .with_tolerance(1E-12)
        .build();
    let mut x = Vector::new(M.rows());

    let start = Instant::now();
    match gs.solve(&M, &b, &mut x) {
        Ok(_) => {
            log::info!(
                "Converged - iteration: {} - residual: {:.2E}",
                gs.iter(),
                gs.residual()
            );
        }
        Err(e) => {
            log::error!("{e}");
        }
    }

    log::info!("Elapsed time: {:.2} sec", start.elapsed().as_secs_f32());

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
