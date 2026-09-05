use super::Matrix;
use crate::{error::Error, linear_algebra::simd};
use rayon::prelude::*;
use std::{
    io::BufRead,
    sync::atomic::{AtomicU32, Ordering},
};

#[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/jm_lib/linear_algebra/matrix/csr.md"))]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Default)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<u32>,
    diag_ptr: Option<Vec<u32>>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub struct CSRMatrixArgs {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<u32>,
    pub diag_ptr: Option<Vec<u32>>,
    pub col_indices: Vec<u32>,
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

    /// Construct a `CSRMatrix` from the coordinates and return it.
    /// Note that this method automatically sets the `diag_ptr`.
    pub fn from_coordinates(
        rows: usize,
        cols: usize,
        mut coordinates: Vec<(u32, u32, f64)>,
    ) -> CSRMatrix {
        let mut sum = 0;
        //
        // coordinate 정렬
        coordinates.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // 1. col_incide, values 배열 세팅
        // 1.1 coordinate에서 col_indices과 values 정보 분리
        let (col_indices, values): (Vec<_>, Vec<_>) = coordinates
            .par_iter()
            .map(|&(_, col, value)| (col, value))
            .unzip();

        // 2. row_ptr 배열 세팅
        let row_ptr_atomic = (0..rows + 1).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();

        // 2.1 행 별로 nnz 카운트 ->
        coordinates
            .par_chunk_by(|a, b| a.0 == b.0)
            .for_each(|chunk| {
                if let Some(&(row, _, _)) = chunk.first() {
                    row_ptr_atomic[row as usize + 1].store(chunk.len() as u32, Ordering::Relaxed);
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

    /// Imports a `CSRMatrix` from a Matrix Market (`.mtx`) file.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The file path cannot be opened.
    /// * The file is empty or missing a valid header block.
    /// * Any data row does not contain exactly 3 valid numbers.
    /// * Parsing integers or floating-point values fails.
    pub fn from_mtx(path: &str) -> Result<Self, Error> {
        log::debug!("Import a CSRMatrix from '{path}'");

        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();
        let mut is_symmetric = false;

        // 1. 헤더 읽기
        let (rows, cols, nnz) = loop {
            let line = lines
                .next()
                .ok_or_else(|| Error::ValueError(format!("Empty file - {}", path)))??;
            let trimmed = line.trim();

            // 주석 및 공백 건너뛰기
            if trimmed.starts_with("%") || trimmed.is_empty() {
                if trimmed.to_lowercase().contains("symmetric") {
                    is_symmetric = true;
                }
                continue;
            }

            let mut tokens = trimmed.split_whitespace();
            if let (Some(r), Some(c), Some(n)) = (tokens.next(), tokens.next(), tokens.next()) {
                let rows = r.parse::<usize>()?;
                let cols = c.parse::<usize>()?;
                let nnz = n.parse::<usize>()?;

                break (rows, cols, nnz);
            } else {
                return Err(Error::ValueError(
                    "The header format is not correct. (needed 3 columns)".into(),
                ));
            }
        };

        // 데이터 메모리 할당 (대칭 행렬일 경우 2배 메모리 할당)
        let mut coordinates = Vec::with_capacity(if is_symmetric { 2 * nnz } else { nnz });

        // 2. 데이터 읽기
        for line in lines {
            let line = line?;
            let trimmed = line.trim();

            // 주석 및 공백 건너뛰기
            if trimmed.starts_with("%") || trimmed.is_empty() {
                continue;
            }

            let mut tokens = trimmed.split_whitespace();

            if let (Some(r), Some(c), Some(v)) = (tokens.next(), tokens.next(), tokens.next()) {
                let row = r.parse::<u32>()? - 1;
                let col = c.parse::<u32>()? - 1;
                let value = v.parse::<f64>()?;

                coordinates.push((row, col, value));

                // 대칭 행렬 처리 (대각 성분이 아닐 때만 반대쪽 추가)
                if is_symmetric && row != col {
                    coordinates.push((col, row, value));
                }
            } else {
                return Err(Error::ValueError(
                    "The data row format is not correct. (needed 3 columns)".into(),
                ));
            }
        }

        Ok(Self::from_coordinates(rows, cols, coordinates))
    }

    /// Return the `row_ptr`.
    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    /// Return the `col_indices`.
    pub fn col_indices(&self) -> &[u32] {
        &self.col_indices
    }

    /// Return the `diag_ptr`.
    pub fn diag_ptr(&self) -> Option<&[u32]> {
        self.diag_ptr.as_deref()
    }

    /// Return the `values`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Set the `diag_ptr` of a `CSRMatrix`.
    pub fn with_diag_ptr(mut self, diag_ptr: Vec<u32>) -> Self {
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
pub fn find_diag_ptr(row_ptr: &[u32], col_indices: &[u32]) -> Result<Vec<u32>, Error> {
    let m = row_ptr.len() - 1;
    let chunk_size = simd::calculate_chunk_size(m);
    let mut diag_ptr = vec![0; m];

    diag_ptr
        .par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, chunk)| {
            for (i, diag) in chunk.iter_mut().enumerate() {
                let global_i = chunk_idx * chunk_size + i;
                let start = row_ptr[global_i] as usize;
                let end = row_ptr[global_i + 1] as usize;

                // global_i 와 동일한 열 인덱스 찾기
                let col_idx = col_indices[start..end]
                    .binary_search(&(global_i as u32))
                    .map_err(|_| Error::MissingDiagonal(global_i))?;

                *diag = (start + col_idx) as u32;
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
        // 대각성분 찾기 기능 검증
        // initialize test
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
        // MTX 포맷 읽기 기능 검증
        // initialize test
        init();

        let path = "resources/mtx/3x3Test.mtx";
        let matrix = CSRMatrix::from_mtx(path)?;

        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.row_ptr, vec![0, 2, 4, 7]);
        assert_eq!(matrix.diag_ptr, Some(vec![0, 2, 6]));
        assert_eq!(matrix.col_indices, vec![0, 2, 1, 2, 0, 1, 2]);
        assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        Ok(())
    }

    #[test]
    fn csr_from_coordinate_test() -> Result<(), Error> {
        // coordinate 배열로부터 생성된 CSRMatrix 검증
        // initialize test
        init();

        let (rows, cols) = (3, 3);
        let coordinates = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (1, 2, 4.0),
            (2, 0, 5.0),
            (2, 1, 6.0),
            (2, 2, 7.0),
        ];

        let M0 = CSRMatrix::from_coordinates(rows, cols, coordinates);
        let M1 = CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr: vec![0, 2, 4, 7],
            diag_ptr: Some(vec![0, 2, 6]),
            col_indices: vec![0, 2, 1, 2, 0, 1, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        });

        assert_eq!(M0, M1);

        Ok(())
    }
}
