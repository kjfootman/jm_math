## Todo

1. Preconditioner 개발
    - [x] Level Scheduling
    - [x] Jacobi
    - [x] SOR
    - [ ] SSOR
    - [x] ILU(0)
    - [ ] ILUT
2. Matrix
    - [x] matrix market (.mtx)파일로부터 `Matrix` 생성
3. MSolver 개발
    - [x] Builder pattern 추가
    - [x] MSolver::Builder() 패턴 추가
    - [x] Builder 패턴에서 preconditioner(&P) P 소유권 가져가도록 수정
    - [ ] Gauss-Seidel 솔버 추가
    - [x] Conjugate-Gradient 솔버 추가
    - [x] immutable로 선언 하도록 수정 (iter, residual 필드 삭제)
    - [x] `PreconType` 추가
    - [x] get_preconditioner() 추가 --> GMRES/HGMRES/CG Preconditioner 선언 간편화
    - [ ] 상태출력 주기 설정 기능 추가 (iteration / write)
    - [ ] SOR preconditioner 검증
4. HGMRES 병렬코드 개선
    - [x] $Pv$ 계산 모듈 개선
        - 병렬모듈(rayon) 적용하였으나 시간이 오래 걸림
        - $(I - 2uu^T)v$ 계산 모듈 개선 필요
        - $v + 2u\sigma$ &nbsp; $\sigma=u^Tv$ 적용
5. jm_math::prelude 모듈에 `msolver` 모듈 추가
    - [x] `Matrix`, `Vector`
    - [x] `MSolver`, ~~`MSolverBuilder`~~, `MyError`, `Preconditioner`
6. MyError
    - [x] `MyError` 필드 String 으로 변경
7. log 기능 추가
    - [x] log 기능 추가
    - [x] dev-dependencies에 log4rs 추가
8. 문서 작업
    - [x] vector.rs
    - [x] matrix.rs
    - [ ] msolver.rs