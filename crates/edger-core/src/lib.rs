//! `edger-core`: the edgeR quasi-likelihood path, ported from edgeR 4.8.2. No R at runtime.

mod apl;
pub mod disp;
pub mod filter;
pub mod glm;
pub mod interp;
mod lapack;
pub mod norm;
pub mod pipeline;
pub mod ql;
mod ql_weights;
pub mod qltest;
