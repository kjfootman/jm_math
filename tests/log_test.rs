use std::error::Error;

#[test]
fn log_test() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();
    let path = std::env::current_dir()?;

    // initialzie log4rs
    log4rs::init_file("log4rs.yaml", Default::default())?;

    // log message
    log::info!("Program started.");
    log::info!("current directory - \"{}\"", path.display());
    log::info!("Program ended.");
    log::info!("elapsed time: {:.2} sec", start.elapsed().as_secs_f32());
    Ok(())
}
