use super::Matrix;
use crate::{error::Error, linear_algebra::simd};
use rayon::prelude::*;
use std::{
    io::BufRead,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    diag_ptr: Option<Vec<usize>>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub struct CSRMatrixArgs {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub diag_ptr: Option<Vec<usize>>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CSRMatrix {
    pub fn from_args(args: CSRMatrixArgs) -> CSRMatrix {
        CSRMatrix {
            rows: args.rows,
            cols: args.cols,
            row_ptr: args.row_ptr,
            diag_ptr: args.diag_ptr,
            col_indices: args.col_indices,
            values: args.values,
        }
    }

    // Construct a `CSRMatrix` from the coordinates and return it.
    pub fn from_coordinates(
        rows: usize,
        cols: usize,
        mut coordinates: Vec<(usize, usize, f64)>,
    ) -> CSRMatrix {
        let mut sum = 0;
        //
        // coordinate 정렬
        coordinates.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // 1. col_incide, values 배열 세팅
        // 1.1 coordinate에서 col_indices과 values 정보 분리
        let (col_indices, values): (Vec<_>, Vec<_>) = coordinates
            .par_iter()
            // mtx 포맷은 1부터 인덱스가 시작
            .map(|&(_, col, value)| (col - 1, value))
            .unzip();

        // 2. row_ptr 배열 세팅
        let row_ptr_atomic = (0..rows + 1)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        // 2.1 행 별로 nnz 카운트 ->
        coordinates
            .par_chunk_by(|a, b| a.0 == b.0)
            .for_each(|chunk| {
                if let Some(&(row, _, _)) = chunk.first() {
                    row_ptr_atomic[row].store(chunk.len(), Ordering::Relaxed);
                }
            });

        // 2.2 row_ptr_atomic 형 변환
        let mut row_ptr = row_ptr_atomic
            .into_par_iter()
            .map(|atomic| atomic.into_inner())
            .collect::<Vec<_>>();

        // 2.3 행 별 nnz 누적 합 -> 최종 row_ptr 완성
        for val in row_ptr.iter_mut() {
            sum += *val;
            *val = sum;
        }

        // 3. 대각 성분 구성
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices).ok();

        CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr,
            diag_ptr,
            col_indices,
            values,
        })
    }

    /// Import a `CSRMatrix` from a MTX file and return it.
    pub fn from_mtx(path: &str) -> Result<Self, Error> {
        log::debug!("Import a CSRMatrix from '{path}'");

        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let lines = reader.lines();
        let mut is_symmetric = false;
        let mut rows = 0;
        let mut cols = 0;
        let mut sum = 0;
        let mut nnz;
        let mut coordinates = Vec::new();

        for line in lines {
            let line = line?;
            let trimmed = line.trim();

            // 주석 및 공백 건너뛰기
            if trimmed.starts_with("%") || trimmed.is_empty() {
                if trimmed.to_lowercase().contains("symmetric") {
                    is_symmetric = true;
                }
                continue;
            }

            let mut tokens = trimmed.split_whitespace();
            let mut next_token = |item| {
                tokens
                    .next()
                    .ok_or_else(|| Error::ValueError(format!("Invalid format: missing {item}")))
            };

            // 헤더 파싱
            if rows == 0 {
                rows = next_token("rows")?.parse::<usize>()?;
                cols = next_token("cols")?.parse::<usize>()?;
                nnz = next_token("nnz")?.parse::<usize>()?;

                coordinates.reserve(nnz);

                log::debug!("rows: {rows} columns: {cols} nnz: {nnz}");
                continue;
            }

            let row = next_token("row")?.parse::<usize>()?;
            let col = next_token("col")?.parse::<usize>()?;
            let value = next_token("value")?.parse::<f64>()?;

            coordinates.push((row, col, value));

            if is_symmetric && row != col {
                coordinates.push((col, row, value));
            }
        }

        // coordinate 정렬
        coordinates.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // 1. col_incide, values 배열 세팅
        // 1.1 coordinate에서 col_indices과 values 정보 분리
        let (col_indices, values): (Vec<_>, Vec<_>) = coordinates
            .par_iter()
            // mtx 포맷은 1부터 인덱스가 시작
            .map(|&(_, col, value)| (col - 1, value))
            .unzip();

        // 2. row_ptr 배열 세팅
        let row_ptr_atomic = (0..rows + 1)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        // 2.1 행 별로 nnz 카운트 ->
        coordinates
            .par_chunk_by(|a, b| a.0 == b.0)
            .for_each(|chunk| {
                if let Some(&(row, _, _)) = chunk.first() {
                    row_ptr_atomic[row].store(chunk.len(), Ordering::Relaxed);
                }
            });

        // 2.2 row_ptr_atomic 형 변환
        let mut row_ptr = row_ptr_atomic
            .into_par_iter()
            .map(|atomic| atomic.into_inner())
            .collect::<Vec<_>>();

        // 2.3 행 별 nnz 누적 합 -> 최종 row_ptr 완성
        for val in row_ptr.iter_mut() {
            sum += *val;
            *val = sum;
        }

        // 3. 대각 성분 구성
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices).ok();

        let matrix = CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr,
            diag_ptr,
            col_indices,
            values,
        });

        Ok(matrix)
    }

    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn diag_ptr(&self) -> Option<&[usize]> {
        self.diag_ptr.as_deref()
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn with_diag_ptr(mut self, diag_ptr: Vec<usize>) -> Self {
        if self.diag_ptr().is_none() {
            self.diag_ptr = Some(diag_ptr)
        }

        self
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

/// Returns pointers to the diagonal elements.
pub fn find_diag_ptr(row_ptr: &[usize], col_indices: &[usize]) -> Result<Vec<usize>, Error> {
    let m = row_ptr.len() - 1;
    let chunk_size = simd::calculate_chunk_size(m);
    let mut diag_ptr = vec![0; m];

    diag_ptr
        .par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, chunk)| {
            for (i, diag) in chunk.iter_mut().enumerate() {
                let global_i = chunk_idx * chunk_size + i;
                let start = row_ptr[global_i];
                let end = row_ptr[global_i + 1];

                // global_i 와 동일한 열 인덱스 찾기
                let col_idx = col_indices[start..end]
                    .binary_search(&global_i)
                    .map_err(|_| Error::MissingDiagonal(global_i))?;

                *diag = start + col_idx;
            }

            Ok::<(), Error>(())
        })?;

    Ok(diag_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn csr_diagonal_test() -> Result<(), Error> {
        init();

        // case1: 대각 성분에 0 이 없을 경우
        log::debug!("Case1 - zero value is not contained in diagonal");
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices)?;

        assert_eq!(diag_ptr, vec![0, 4, 6, 8]);

        // case2: 대각 성분에 0 이 있을 경우
        log::debug!("Case2 - zero value contained in diagonal");
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 1, 3, 3];
        let diag_ptr = find_diag_ptr(&row_ptr, &col_indices).inspect_err(|e| log::warn!("{e}"));

        assert!(diag_ptr.is_err());

        Ok(())
    }

    #[test]
    fn csr_from_mtx_test() -> Result<(), Error> {
        init();
        let path = "resources/mtx/3x3Test.mtx";
        let matrix = CSRMatrix::from_mtx(path)?;

        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.row_ptr, vec![0, 2, 4, 7]);
        assert_eq!(matrix.col_indices, vec![0, 2, 1, 2, 0, 1, 2]);
        assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        Ok(())
    }
}
