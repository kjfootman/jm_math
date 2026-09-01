# TODO

## 코어 개발

- [x] MKT 파일로터 `CRSMatrix` 행렬 불러오기
- [x] MTK 파일로부터 `Vector` 불러오기
- [x] CSRMatrix에 diag_ptr 필드 추가
- [x] CSRMatrix에 with_diag_ptr 매소드 개발
- [x] Simple coordinate로부터 CSRMatrix 생성 -> CSRMatrix::from_coo
- [x] CSRMatrix row_ptr, col_indices, diag_ptr Vec<u32> 타입으로 수정
- [ ] DenseMatrix 개발
  - [ ] Matrix 트레이트 구현

## 솔버 개발

- [ ] SOR(w) 솔버 개발
- [ ] GMRES 솔버 개발
  - [ ] Given's rotation 개발
  - [ ] HGMRES 솔버 개발
- [ ] Conjugate Gradient 솔버 개발
- [x] Gauss-Seidel 솔버 bound check 해제

## Preconditioner 개발

- [ ] ILU Preconditioner 개발
- [ ] SOR(w) Preconditioner 개발

- [ ] SPMV 연산 bound check 해제

- [ ] Level scheduling 기법 개발
