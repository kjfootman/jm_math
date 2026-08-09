use std::io::BufRead;

use super::Matrix;
use crate::{error::Error, linear_algebra::simd};
use rayon::prelude::*;

#[derive(Debug, Default)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    // diag_ptr: Option<Vec<usize>>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub struct CSRMatrixArgs {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    // pub diag_ptr: Option<Vec<usize>>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CSRMatrix {
    pub fn from_args(args: CSRMatrixArgs) -> CSRMatrix {
        CSRMatrix {
            rows: args.rows,
            cols: args.cols,
            row_ptr: args.row_ptr,
            // diag_ptr: args.diag_ptr,
            col_indices: args.col_indices,
            values: args.values,
        }
    }

    pub fn from_mtx(path: &str) -> Result<Self, Error> {
        log::info!("Import CSRMatrix from '{path}'");
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();
        let mut rows = 0;
        let mut cols = 0;
        let mut nnz = 0;
        let mut row_ptr = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        while let Some(line) = lines.next() {
            let line = line?;
            let trimmed = line.trim();

            // 주석 건너뛰기
            if trimmed.starts_with("%") {
                continue;
            }

            let tokens = trimmed.split_whitespace().collect::<Vec<_>>();

            // 헤더 파싱
            if rows == 0 {
                rows = tokens[0]
                    .parse::<usize>()
                    .map_err(|e| Error::Parse(e.to_string()))?;
                cols = tokens[1]
                    .parse::<usize>()
                    .map_err(|e| Error::Parse(e.to_string()))?;
                nnz = tokens[2]
                    .parse::<usize>()
                    .map_err(|e| Error::Parse(e.to_string()))?;

                row_ptr.reserve(rows + 1);
                col_indices.reserve(nnz);
                values.reserve(nnz);

                continue;
            }

            if tokens.len() >= 3 {
                let row = tokens[0]
                    .parse::<usize>()
                    .map_err(|e| Error::Parse(e.to_string()))?;
                let col = tokens[1]
                    .parse::<usize>()
                    .map_err(|e| Error::Parse(e.to_string()))?;
                let value = tokens[2]
                    .parse::<f64>()
                    .map_err(|e| Error::Parse(e.to_string()))?;

                row_ptr.push(row);
                col_indices.push(col);
                values.push(value)
            }

            // println!("{rows}, {cols}, {nnz}");
        }

        println!("{rows}, {cols}, {nnz}");

        todo!()
    }

    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    // pub fn diag_ptr(&self) -> Option<&[usize]> {
    //     self.diag_ptr.as_deref()
    // }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl Matrix for CSRMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }
}

/// Returns pointers to the diagonal elements
pub fn find_diag_ptr(row_ptr: &[usize], col_indices: &[usize]) -> Result<Vec<usize>, Error> {
    let m = row_ptr.len() - 1;
    let chunk_size = simd::calculate_chunk_size(m);
    let mut diag_ptr = vec![0; m];

    diag_ptr
        .par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, local_diag_ptr)| {
            for (i, diag) in local_diag_ptr.iter_mut().enumerate() {
                let global_i = chunk_idx * chunk_size + i;
                let start = row_ptr[global_i];
                let end = row_ptr[global_i + 1];

                match col_indices[start..end].binary_search(&global_i) {
                    Ok(value) => *diag = start + value,
                    Err(_) => {
                        let msg = format!(
                            "Failed to find the pointer to the diagonal element of row {}",
                            global_i
                        );
                        return Err(Error::ValueError(msg));
                    }
                }
            }

            Ok(())
        })?;

    Ok(diag_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn csr_diagonal_test() -> Result<(), Error> {
        // case1: 대각 성분에 0 이 없을 경우
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices)?;

        assert_eq!(diag_ptr, vec![0, 4, 6, 8]);

        // case2: 대각 성분에 0 이 있을 경우
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 1, 3, 3];
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices).inspect_err(|e| println!("{e:#?}"));

        assert!(diag_ptr.is_err());

        Ok(())
    }

    #[test]
    fn csr_from_mtx_test() -> Result<(), Error> {
        init();
        // log::info!("Current directory: {}", env::current_dir()?.display());

        let path = "resources/mtx/e40r5000.mtx";
        let matrix = CSRMatrix::from_mtx(path)?;

        Ok(())
    }
}
