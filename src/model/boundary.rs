// Rust guideline compliant 2026-04-03
//! GLiNER2.5 boundary-prediction head.
//!
//! This module implements the GLiNER2.5 `boundary_head` and companion modules
//! (`relation_scorer`, `record_decoder`) as observed in the released
//! checkpoints (`fastino/gliner2.5-{small,base,multi}-v1`).
//!
//! Unlike GLiNER2's span-enumeration path (span_rep + classifier over a
//! `(seq_len, max_width)` grid), the boundary architecture scores entity
//! start/end/inside positions per schema query, proposes sparse candidate
//! spans, reranks them, and shares the candidate pool with relation scoring.
//!
//! # Module layout (safetensors keys)
//!
//! ```text
//! boundary_head.boundary_encoder.{left,right}_projection, layer_norm,
//!     attention_blocks.{0,1}, refinement_blocks.0, output_projection,
//!     bos_state, eos_state
//! boundary_head.boundary_query_head.{start,end,inside}_query_projection,
//!     {start,end}_boundary_projection, inside_text_projection
//! boundary_head.boundary_proposer.start_query_projection,
//!     start_pair_projection, end_key_projection
//! boundary_head.shared_pool_builder.{start,end}_projection
//! boundary_head.shared_pool_scorer.*
//! boundary_head.pair_scorer.*
//! boundary_head.count_head, boundary_head.null_projection,
//! boundary_head.candidate_encoder
//! classifier.{0,3}                      (classification head)
//! relation_scorer.*                     (joint IE)
//! record_decoder.*                      (structured records)
//! ```
//!
//! Forward-pass decoding lives in `crate::inference`; this module owns the
//! parameter structures and weight loading.

use candle_core::{Device, DType, Tensor};
use candle_nn::{layer_norm, linear, Linear, Module, VarBuilder};

use crate::config::{Architecture, BoundaryConfig, ExtractorConfig};
use crate::error::{GlinerError, Result};

/// LayerNorm epsilon used by the GLiNER2.5 boundary heads.
const BOUNDARY_LAYER_NORM_EPS: f64 = 1e-5;

/// Finite sentinel used by upstream instead of `-inf` (fp16-safe).
pub const MASK_LOGIT: f32 = -1e4;

/// Local attention window of the boundary self-attention blocks
/// (`boundary_attention_window`, 0 disables windowing).
const BOUNDARY_ATTENTION_WINDOW: usize = 128;

/// Extract a 2D matrix from a weight tensor as flat row-major data.
fn mat2(t: &Tensor) -> Result<(Vec<f32>, usize, usize)> {
    let dims = t.dims();
    if dims.len() == 1 {
        let v = t.to_vec1().map_err(|e| GlinerError::inference(format!("{e}")))?;
        return Ok((v, dims[0], 1));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let v = t.to_vec2().map_err(|e| GlinerError::inference(format!("{e}")))?;
    let mut flat = Vec::with_capacity(rows * cols);
    for row in v {
        flat.extend_from_slice(&row);
    }
    Ok((flat, rows, cols))
}

/// Dense projection applied to a single feature vector: y = Wx + b.
fn apply_linear(w: &[f32], w_cols: usize, b: &[f32], x: &[f32]) -> Vec<f32> {
    let rows = w.len() / w_cols;
    let mut out = vec![0.0f32; rows];
    for (r, o) in out.iter_mut().enumerate() {
        let wrow = &w[r * w_cols..(r + 1) * w_cols];
        let mut acc = b[r];
        for c in 0..w_cols {
            acc += wrow[c] * x[c];
        }
        *o = acc;
    }
    out
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Stable descending sort returning original indices.
fn argsort_desc_stable(values: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Extract a linear bias as a flat vector (zeros when absent).
fn bias1(bias: Option<&Tensor>) -> Result<Vec<f32>> {
    match bias {
        Some(t) => t.to_vec1().map_err(|e| GlinerError::inference(format!("{e}"))),
        None => Ok(Vec::new()),
    }
}

/// LayerNorm (weight, bias) parameter vectors from a candle LayerNorm.
fn ln_params(ln: &candle_nn::LayerNorm) -> Result<(Vec<f32>, Vec<f32>)> {
    let w = ln.weight().to_vec1().map_err(|e| GlinerError::inference(format!("{e}")))?;
    let b = match ln.bias() { Some(t) => t.to_vec1().map_err(|e| GlinerError::inference(format!("{e}")))?, None => vec![0.0; w.len()] };
    Ok((w, b))
}


/// Self-attention block inside the boundary encoder.
#[derive(Debug)]
struct AttentionBlock {
    norm: candle_nn::LayerNorm,
    qkv_projection: Linear,
    output_projection: Linear,
    num_heads: usize,
}

impl AttentionBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(dim, BOUNDARY_LAYER_NORM_EPS, vb.pp("norm"))?,
            qkv_projection: linear(dim, 3 * dim, vb.pp("qkv_projection"))?,
            output_projection: linear(dim, dim, vb.pp("output_projection"))?,
            num_heads,
        })
    }

    /// Pre-norm multi-head self-attention over boundary states with a local
    /// window and validity mask.
    fn forward(&self, x: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        let normed = self.norm.forward(x)?;
        let qkv = self.qkv_projection.forward(&normed)?;
        let head_dim = dim / self.num_heads;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?.contiguous()?;
        let (q, k, v) = (
            qkv.get(0)?,
            qkv.get(1)?,
            qkv.get(2)?,
        );
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;

        // Allow keys that are valid boundaries, within the local window of
        // the query, or on the diagonal (keeps padded rows NaN-free).
        let window = BOUNDARY_ATTENTION_WINDOW;
        let mask_flat = key_mask
            .flatten_all()
            .map_err(|e| GlinerError::inference(format!("{e}")))?
            .to_vec1::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        let mut bias = vec![0.0f32; batch * seq_len * seq_len];
        for b in 0..batch {
            let n_b_valid = |ki: usize| mask_flat[b * seq_len + ki];
            for qi in 0..seq_len {
                for ki in 0..seq_len {
                    if n_b_valid(ki) == 0.0 && ki != qi {
                        bias[(b * seq_len + qi) * seq_len + ki] = MASK_LOGIT;
                    } else if window > 0 && (qi as i64 - ki as i64).abs() > window as i64 {
                        bias[(b * seq_len + qi) * seq_len + ki] = MASK_LOGIT;
                    }
                }
            }
        }
        let bias =
            Tensor::from_slice(&bias, (batch, 1, seq_len, seq_len), x.device()).map_err(
                |e| GlinerError::inference(format!("bias tensor failed: {e}")),
            )?;
        let shape = attn.shape().clone();
        let attn = (attn + bias.broadcast_as(&shape)?)?;

        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((batch, seq_len, dim))?;
        let out = self.output_projection.forward(&out)?;
        // Residual connection + mask padding boundaries
        let masked = (x + out)?;
        // key_mask is [B, 1, 1, N]; squeeze to [B, N] then unsqueeze to [B, N, 1]
        let m2d = key_mask
            .squeeze(1)
            .map_err(|e| GlinerError::inference(format!("{e}")))?
            .squeeze(1)
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        let m3d = m2d
            .unsqueeze(2)
            .map_err(|e| GlinerError::inference(format!("{e}")))?
            .broadcast_as(masked.shape())
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        Ok((masked * m3d)?)
    }
}

/// Refinement (FFN) block inside the boundary encoder.
#[derive(Debug)]
struct RefinementBlock {
    input_projection: Linear,
    norm: candle_nn::LayerNorm,
    output_projection: Linear,
}

impl RefinementBlock {
    fn load(vb: VarBuilder, boundary_dim: usize) -> Result<Self> {
        Ok(Self {
            input_projection: linear(boundary_dim, 4 * boundary_dim, vb.pp("input_projection"))?,
            norm: layer_norm(boundary_dim, BOUNDARY_LAYER_NORM_EPS, vb.pp("norm"))?,
            output_projection: linear(
                2 * boundary_dim,
                boundary_dim,
                vb.pp("output_projection"),
            )?,
        })
    }

    /// Pre-norm SwiGLU refinement: split the expanded projection into
    /// value/gate halves and gate with SiLU before the output projection.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_projection.forward(&self.norm.forward(x)?)?;
        let dims = normed.dims();
        let half = dims[dims.len() - 1] / 2;
        let value = normed.narrow(candle_core::D::Minus1, 0, half)?;
        let gate = normed.narrow(candle_core::D::Minus1, half, half)?;
        let update = (value * candle_nn::ops::silu(&gate)?)?;
        let out = self.output_projection.forward(&update)?;
        Ok((x + out)?)
    }
}

/// Boundary encoder: projects token embeddings into boundary states, refines
/// them with local self-attention, and produces paired left/right features.
#[derive(Debug)]
pub struct BoundaryEncoder {
    left_projection: Linear,
    right_projection: Linear,
    layer_norm: candle_nn::LayerNorm,
    attention_blocks: Vec<AttentionBlock>,
    refinement_blocks: Vec<RefinementBlock>,
    output_projection: Linear,
    bos_state: Tensor,
    eos_state: Tensor,
}

impl BoundaryEncoder {
    /// Load from `vb` rooted at `boundary_head.boundary_encoder`.
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        cfg: &BoundaryConfig,
        device: &Device,
    ) -> Result<Self> {
        const ATTENTION_LAYERS: usize = 2;
        const REFINEMENT_LAYERS: usize = 1;
        const ATTENTION_HEADS: usize = 4;

        let mut attention_blocks = Vec::with_capacity(ATTENTION_LAYERS);
        for i in 0..ATTENTION_LAYERS {
            attention_blocks.push(AttentionBlock::load(
                vb.pp(format!("attention_blocks.{i}")),
                cfg.boundary_dim,
                ATTENTION_HEADS,
            )?);
        }

        let mut refinement_blocks = Vec::with_capacity(REFINEMENT_LAYERS);
        for i in 0..REFINEMENT_LAYERS {
            refinement_blocks.push(RefinementBlock::load(
                vb.pp(format!("refinement_blocks.{i}")),
                cfg.boundary_dim,
            )?);
        }

        Ok(Self {
            left_projection: linear(hidden_size, cfg.boundary_dim, vb.pp("left_projection"))?,
            right_projection: linear(hidden_size, cfg.boundary_dim, vb.pp("right_projection"))?,
            layer_norm: layer_norm(
                cfg.boundary_dim,
                BOUNDARY_LAYER_NORM_EPS,
                vb.pp("layer_norm"),
            )?,
            attention_blocks,
            refinement_blocks,
            output_projection: linear(
                2 * cfg.boundary_dim,
                cfg.boundary_dim,
                vb.pp("output_projection"),
            )?,
            bos_state: vb.get((hidden_size,), "bos_state")?,
            eos_state: vb.get((hidden_size,), "eos_state")?.to_device(device)?,
        })
    }

    /// Compute boundary states from word-level text states.
    ///
    /// Implements the upstream `BoundaryEncoder`: boundaries sit between
    /// tokens, so sample `b` with `n_b` valid words yields `n_b + 1`
    /// boundaries; boundary `i` sees `[BOS, w0, .., w(n-1), EOS]` shifted
    /// left/right.
    ///
    /// * `text_states`: `(batch, L, hidden_size)`
    /// * `text_lengths`: valid word count per sample
    ///
    /// Returns `(batch, N=L+1, boundary_dim)`.
    pub fn forward(&self, text_states: &Tensor, text_lengths: &[usize]) -> Result<Tensor> {
        let (batch, seq_len, _) = text_states.dims3()?;
        let n = seq_len + 1;
        let h = self.bos_state.dims()[0];

        // left[i] = BOS at i=0 else token i-1 ; right[i] = token i if i < n_b
        // else EOS.
        let mut left = Vec::with_capacity(batch * n * h);
        let mut right = Vec::with_capacity(batch * n * h);
        let flat = text_states
            .reshape((batch * seq_len, h))
            .map_err(|e| GlinerError::inference(format!("reshape failed: {e}")))?
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("gather failed: {e}")))?;
        let bos = self.bos_state.to_vec1::<f32>().map_err(|e| {
            GlinerError::inference(format!("bos_state read failed: {e}"))
        })?;
        let eos = self.eos_state.to_vec1::<f32>().map_err(|e| {
            GlinerError::inference(format!("eos_state read failed: {e}"))
        })?;

        for b in 0..batch {
            let n_b = text_lengths[b].min(seq_len);
            for i in 0..n {
                if i == 0 {
                    left.extend_from_slice(&bos);
                } else {
                    left.extend_from_slice(&flat[b * seq_len + i - 1]);
                }
                if i >= n_b {
                    right.extend_from_slice(&eos);
                } else {
                    right.extend_from_slice(&flat[b * seq_len + i]);
                }
            }
        }

        let left = Tensor::from_slice(&left, (batch, n, h), text_states.device())
            .map_err(|e| GlinerError::inference(format!("left tensor failed: {e}")))?;
        let right = Tensor::from_slice(&right, (batch, n, h), text_states.device())
            .map_err(|e| GlinerError::inference(format!("right tensor failed: {e}")))?;

        let lp = self.left_projection.forward(&left)?;
        let rp = self.right_projection.forward(&right)?;
        let combined = Tensor::cat(&[lp, rp], candle_core::D::Minus1)?;
        let mut state = self.layer_norm.forward(&self.output_projection.forward(&combined)?)?;

        // Boundary validity mask: boundary i valid iff i <= n_b.
        let mut mask_rows = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let n_b = text_lengths[b].min(seq_len);
            for i in 0..n {
                mask_rows.push(if i <= n_b { 1.0f32 } else { 0.0 });
            }
        }
        let mask =
            Tensor::from_slice(&mask_rows, (batch, 1, 1, n), text_states.device()).map_err(
                |e| GlinerError::inference(format!("mask tensor failed: {e}")),
            )?;

        for block in &self.attention_blocks {
            state = block.forward(&state, &mask)?;
        }
        for block in &self.refinement_blocks {
            state = block.forward(&state)?;
        }

        let state_dim = state.dims()[2];
        let state_mask = mask
            .reshape((batch, n, 1))
            .map_err(|e| GlinerError::inference(format!("reshape mask: {e}")))?
            .broadcast_as((batch, n, state_dim))
            .map_err(|e| GlinerError::inference(format!("broadcast mask: {e}")))?;
        Ok((state * state_mask)?)
    }

    /// The learned [BOS] token embedding used to anchor proposals.
    pub fn bos_state(&self) -> &Tensor {
        &self.bos_state
    }

    /// The learned [EOS] token embedding used to anchor proposals.
    pub fn eos_state(&self) -> &Tensor {
        &self.eos_state
    }
}

/// Per-query projections producing start/end/inside boundary evidence.
#[derive(Debug)]
pub struct BoundaryQueryHead {
    start_query_projection: Linear,
    end_query_projection: Linear,
    inside_query_projection: Linear,
    start_boundary_projection: Linear,
    end_boundary_projection: Linear,
    inside_text_projection: Linear,
}

impl BoundaryQueryHead {
    /// Load from `vb` rooted at `boundary_head.boundary_query_head`.
    pub fn load(vb: VarBuilder, hidden_size: usize, cfg: &BoundaryConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        Ok(Self {
            start_query_projection: linear(hidden_size, d, vb.pp("start_query_projection"))?,
            end_query_projection: linear(hidden_size, d, vb.pp("end_query_projection"))?,
            inside_query_projection: linear(hidden_size, d, vb.pp("inside_query_projection"))?,
            start_boundary_projection: linear(d, d, vb.pp("start_boundary_projection"))?,
            end_boundary_projection: linear(d, d, vb.pp("end_boundary_projection"))?,
            inside_text_projection: linear(hidden_size, d, vb.pp("inside_text_projection"))?,
        })
    }

    /// Project each query embedding into boundary query space.
    pub fn project_queries(&self, queries: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.start_query_projection.forward(queries)?,
            self.end_query_projection.forward(queries)?,
            self.inside_query_projection.forward(queries)?,
        ))
    }

    /// Project boundary states / token embeddings into key space.
    pub fn project_keys(
        &self,
        boundary_states: &Tensor,
        token_embs: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.start_boundary_projection.forward(boundary_states)?,
            self.end_boundary_projection.forward(boundary_states)?,
            self.inside_text_projection.forward(token_embs)?,
        ))
    }
}

/// Sparse proposal stage: pairs top-k starts with ends without distance
/// restrictions.
#[derive(Debug)]
pub struct BoundaryProposer {
    start_query_projection: Linear,
    start_pair_projection: Linear,
    end_key_projection: Linear,
}

impl BoundaryProposer {
    /// Load from `vb` rooted at `boundary_head.boundary_proposer`.
    pub fn load(vb: VarBuilder, hidden_size: usize, d: usize) -> Result<Self> {
        Ok(Self {
            start_query_projection: linear(hidden_size, d / 2, vb.pp("start_query_projection"))?,
            start_pair_projection: linear(d, d, vb.pp("start_pair_projection"))?,
            end_key_projection: linear(d, d, vb.pp("end_key_projection"))?,
        })
    }
}

/// Builds the shared candidate pool from proposed boundaries.
#[derive(Debug)]
pub struct SharedPoolBuilder {
    start_projection: Linear,
    end_projection: Linear,
}

impl SharedPoolBuilder {
    /// Load from `vb` rooted at `boundary_head.shared_pool_builder`.
    pub fn load(vb: VarBuilder, d: usize) -> Result<Self> {
        Ok(Self {
            start_projection: linear(d, d, vb.pp("start_projection"))?,
            end_projection: linear(d, d, vb.pp("end_projection"))?,
        })
    }
}

/// Scores candidates drawn from the shared pool against each query.
#[derive(Debug)]
pub struct SharedPoolScorer {
    query_projection: Linear,
    start_projection: Linear,
    end_projection: Linear,
    content_projection: Linear,
    length_projection: Linear,
    prior_projection: Linear,
    film: Linear,
    film_output_hidden: Linear,
    film_output_out: Linear,
    candidate_norm: candle_nn::LayerNorm,
}

impl SharedPoolScorer {
    /// Load from `vb` rooted at `boundary_head.shared_pool_scorer`.
    pub fn load(vb: VarBuilder, hidden_size: usize, cfg: &BoundaryConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        let c = cfg.content_dim;
        Ok(Self {
            query_projection: linear(hidden_size, d, vb.pp("query_projection"))?,
            start_projection: linear(d, d, vb.pp("start_projection"))?,
            end_projection: linear(d, d, vb.pp("end_projection"))?,
            content_projection: linear(c, d, vb.pp("content_projection"))?,
            length_projection: linear(3, d, vb.pp("length_projection"))?,
            prior_projection: linear(1, d, vb.pp("prior_projection"))?,
            film: linear(d, 2 * d, vb.pp("film"))?,
            film_output_hidden: linear(d, c, vb.pp("film_output.0"))?,
            film_output_out: linear(c, 1, vb.pp("film_output.3"))?,
            candidate_norm: layer_norm(d, BOUNDARY_LAYER_NORM_EPS, vb.pp("candidate_norm"))?,
        })
    }

    /// Candidate LayerNorm parameters (`candidate_norm`).
    pub fn candidate_norm_params(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        ln_params(&self.candidate_norm)
    }
}

/// Pairwise reranking scorer combining endpoint compat, content evidence and
/// inside-score weighting.
#[derive(Debug)]
pub struct PairScorer {
    start_endpoint_projection: Linear,
    end_endpoint_projection: Linear,
    endpoint_difference_projection: Linear,
    inside_weight: Linear,
    content_bias: Linear,
    compat_mix: Linear,
    length_query_projection: Linear,
    query_gate: Linear,
    content_layer_norm: candle_nn::LayerNorm,
    content_value_projection: Linear,
}

impl PairScorer {
    /// Load from `vb` rooted at `boundary_head.pair_scorer`.
    pub fn load(vb: VarBuilder, hidden_size: usize, cfg: &BoundaryConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        let c = cfg.content_dim;
        Ok(Self {
            start_endpoint_projection: linear(d, d, vb.pp("start_endpoint_projection"))?,
            end_endpoint_projection: linear(d, d, vb.pp("end_endpoint_projection"))?,
            endpoint_difference_projection: linear(2 * d, 1, vb.pp("endpoint_difference_projection"))?,
            inside_weight: linear(hidden_size, 1, vb.pp("inside_weight"))?,
            content_bias: linear(c, 1, vb.pp("content_bias"))?,
            compat_mix: linear(cfg.pair_dim / 16, 1, vb.pp("compat_mix"))?,
            length_query_projection: linear(hidden_size, 3, vb.pp("length_query_projection"))?,
            query_gate: linear(hidden_size, c, vb.pp("query_gate"))?,
            content_layer_norm: layer_norm(c, BOUNDARY_LAYER_NORM_EPS, vb.pp("content_pooler.layer_norm"))?,
            content_value_projection: linear(hidden_size, c, vb.pp("content_pooler.value_projection"))?,
        })
    }

    /// Span-content value-projection weights (`content_pooler.value_projection`).
    pub fn content_value_weight(&self) -> Tensor {
        self.content_value_projection.weight().clone()
    }

    /// Span-content value-projection bias.
    pub fn content_value_bias(&self) -> Option<Tensor> {
        self.content_value_projection.bias().cloned()
    }

    /// Span-content LayerNorm parameters (`content_pooler.layer_norm`).
    pub fn content_ln_params(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        ln_params(&self.content_layer_norm)
    }
}

/// Classification head of GLiNER2.5 (`classifier.{0,3}` at the root).
///
/// Note the index shift vs GLiNER2's `classifier.{0,2}`.
#[derive(Debug)]
pub struct BoundaryClassifier {
    hidden: Linear,
    out: Linear,
}

impl BoundaryClassifier {
    /// Load from `vb` rooted at `classifier`.
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            hidden: linear(hidden_size, 2 * hidden_size, vb.pp("0"))?,
            out: linear(2 * hidden_size, 1, vb.pp("3"))?,
        })
    }

    /// Score query embeddings: `(num_queries, hidden) -> (num_queries, 1)`.
    pub fn forward(&self, queries: &Tensor) -> Result<Tensor> {
        let h = self.hidden.forward(queries)?.relu()?;
        Ok(self.out.forward(&h)?)
    }
}

/// Relation scoring module for joint entity/relation extraction
/// (`relation_scorer.*` at the root).
#[derive(Debug)]
pub struct RelationScorer {
    head_content_projection: Linear,
    tail_content_projection: Linear,
    relation_content_gate: Linear,
    mlp_hidden: Linear,
    mlp_out: Linear,
    content_linear: Linear,
}

impl RelationScorer {
    /// Load from `vb` rooted at `relation_scorer`.
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            head_content_projection: linear(
                hidden_size,
                hidden_size,
                vb.pp("head_content_projection"),
            )?,
            tail_content_projection: linear(
                hidden_size,
                hidden_size,
                vb.pp("tail_content_projection"),
            )?,
            relation_content_gate: linear(
                2 * hidden_size,
                hidden_size,
                vb.pp("relation_content_gate"),
            )?,
            // mlp.0 input (observed 4610 @ h=768): 6h concat of projected
            // head/tail states + gated content + raw spans + 2 scalar features
            mlp_hidden: linear(6 * hidden_size + 2, hidden_size, vb.pp("mlp.0"))?,
            mlp_out: linear(hidden_size, 1, vb.pp("mlp.3"))?,
            // content_linear input: head/tail projected concat
            content_linear: linear(4 * hidden_size, 1, vb.pp("content_linear"))?,
        })
    }

    /// Score relation pairs from boundary endpoint states.
    ///
    /// Pure-Rust forward pass mirroring `SparseRelationScorer.forward()`.
    ///
    /// * `boundary_states` – `[L+1][d]` from `BoundaryEncoder`
    /// * `relation_query` – `[2*hidden]` directional relation query state
    /// * `head_spans`, `tail_spans` – half-open `(start, end)` per pair
    /// * `text_len` – valid sequence length for distance normalisation
    ///
    /// Returns one logit per pair.
    pub fn forward(
        &self,
        boundary_states: &[Vec<f32>],
        relation_query: &[f32],
        head_spans: &[(usize, usize)],
        tail_spans: &[(usize, usize)],
        text_len: usize,
    ) -> Result<Vec<f32>> {
        let h = self.head_content_projection.weight().dims()[0]; // hidden_size
        let n_pairs = head_spans.len();
        if n_pairs == 0 {
            return Ok(Vec::new());
        }
        let d = boundary_states[0].len(); // boundary_dim
        let seq_len = boundary_states.len(); // L+1

        // Weight extraction
        let (w_hcp, _, _) = mat2(&self.head_content_projection.weight())?;
        let b_hcp = bias1(self.head_content_projection.bias())?;
        let (w_tcp, _, _) = mat2(&self.tail_content_projection.weight())?;
        let b_tcp = bias1(self.tail_content_projection.bias())?;
        let (w_rcg, _, rcg_in) = mat2(&self.relation_content_gate.weight())?;
        let b_rcg = bias1(self.relation_content_gate.bias())?;
        let (w_mlp0, _, mlp0_in) = mat2(&self.mlp_hidden.weight())?;
        let b_mlp0 = bias1(self.mlp_hidden.bias())?;
        let (w_mlp3, _, _) = mat2(&self.mlp_out.weight())?;
        let b_mlp3 = bias1(self.mlp_out.bias())?;
        let (w_cl, _, cl_in) = mat2(&self.content_linear.weight())?;
        let b_cl = bias1(self.content_linear.bias())?;

        let gather = |pos: usize| -> Vec<f32> {
            let safe = pos.min(seq_len - 1);
            boundary_states[safe].clone()
        };

        let mut scores = Vec::with_capacity(n_pairs);
        for i in 0..n_pairs {
            let (hs, he) = head_spans[i];
            let (ts, te) = tail_spans[i];

            let h_start_st = gather(hs);
            let h_end_st = gather(he.saturating_sub(1).min(seq_len - 1));
            let t_start_st = gather(ts);
            let t_end_st = gather(te.saturating_sub(1).min(seq_len - 1));

            // Positional features
            let delta = (ts as f32) - (hs as f32);
            let order = delta.signum();
            let dist = delta.abs() / (text_len.max(1) as f32);

            // MLP features: [h_start, h_end, t_start, t_end, rel, order, dist]
            let mut feats = Vec::with_capacity(6 * h + 2);
            feats.extend_from_slice(&h_start_st);
            feats.extend_from_slice(&h_end_st);
            feats.extend_from_slice(&t_start_st);
            feats.extend_from_slice(&t_end_st);
            feats.extend_from_slice(relation_query);
            feats.push(order);
            feats.push(dist);

            let mlp_hidden = apply_linear(&w_mlp0, mlp0_in, &b_mlp0, &feats);
            let mlp_hidden_gelu: Vec<f32> = mlp_hidden.iter().map(|v| gelu_f32(*v)).collect();
            let mut score = apply_linear(&w_mlp3, h, &b_mlp3, &mlp_hidden_gelu)[0];

            // Biaffine content path
            // Pool span as mean of boundary states in [start, end)
            let pool = |start: usize, end: usize| -> Vec<f32> {
                let s = start.min(seq_len - 1);
                let e = end.min(seq_len);
                let width = (e - s).max(1);
                let mut result = vec![0.0f32; d];
                for pos in s..e {
                    for j in 0..d {
                        result[j] += boundary_states[pos][j];
                    }
                }
                for v in &mut result {
                    *v /= width as f32;
                }
                result
            };

            let head_content_raw = pool(hs, he.max(hs + 1));
            let head_content = apply_linear(&w_hcp, h, &b_hcp, &head_content_raw);
            let tail_content_raw = pool(ts, te.max(ts + 1));
            let tail_content = apply_linear(&w_tcp, h, &b_tcp, &tail_content_raw);

            // gate = sigmoid(relation_content_gate(rel))
            let gate_raw = apply_linear(&w_rcg, rcg_in, &b_rcg, relation_query);
            let gate: Vec<f32> = gate_raw.iter().map(|v| sigmoid_f32(*v)).collect();

            // biaffine = sum(head_content * gate * tail_content) / sqrt(H)
            let biaffine: f32 = head_content
                .iter()
                .zip(&gate)
                .zip(&tail_content)
                .map(|((&hc, &g), &tc)| hc * g * tc)
                .sum::<f32>()
                / (h as f32).sqrt();

            // content_linear = linear(cat(head_content, tail_content, rel))
            let mut cl_feats = Vec::with_capacity(2 * h + relation_query.len());
            cl_feats.extend_from_slice(&head_content);
            cl_feats.extend_from_slice(&tail_content);
            cl_feats.extend_from_slice(relation_query);
            let cl_out = apply_linear(&w_cl, cl_in, &b_cl, &cl_feats);
            let linear_term = cl_out[0];

            score += biaffine + linear_term;
            scores.push(score);
        }
        Ok(scores)
    }
}

/// Structured-record decoder (`record_decoder.*` at the root).
#[derive(Debug)]
pub struct RecordDecoder {
    hidden_size: usize,
    record_dim: usize,
    cand_proj: Linear,
    field_proj: Linear,
    inst_proj: Linear,
    k_proj: Linear,
    q_proj: Linear,
    v_proj: Linear,
    instance_embed: Tensor,
    null_embed: Tensor,
    latent_seed_head: Linear,
    object_head: Linear,
}

/// One decoded record: field query id → selected (start, end) spans.
#[derive(Debug, Clone)]
pub struct DecodedRecord {
    /// field_query_id → list of (start, end) half-open spans.
    pub fields: std::collections::HashMap<usize, Vec<(usize, usize)>>,
    /// Per-field scores.
    pub field_scores: std::collections::HashMap<usize, Vec<f32>>,
    /// Overall record confidence score.
    pub score: f32,
}

impl RecordDecoder {
    /// Number of learned record-instance queries.
    pub const NUM_INSTANCE_QUERIES: usize = 32;

    /// Load from `vb` rooted at `record_decoder`.
    pub fn load(vb: VarBuilder, hidden_size: usize, device: &Device) -> Result<Self> {
        let record_dim = 128;
        Ok(Self {
            hidden_size,
            record_dim,
            cand_proj: linear(hidden_size, record_dim, vb.pp("cand_proj"))?,
            field_proj: linear(hidden_size, record_dim, vb.pp("field_proj"))?,
            inst_proj: linear(hidden_size, record_dim, vb.pp("inst_proj"))?,
            k_proj: linear(hidden_size, record_dim, vb.pp("k_proj"))?,
            q_proj: linear(hidden_size, record_dim, vb.pp("q_proj"))?,
            v_proj: linear(hidden_size, hidden_size, vb.pp("v_proj"))?,
            instance_embed: vb
                .get((Self::NUM_INSTANCE_QUERIES, hidden_size), "instance_embed")?
                .to_device(device)?,
            null_embed: vb.get((record_dim,), "null_embed")?.to_device(device)?,
            latent_seed_head: linear(hidden_size, 1, vb.pp("latent_seed_head"))?,
            object_head: linear(hidden_size, 1, vb.pp("object_head"))?,
        })
    }

    /// Anchorless-mode forward: learned instance queries cross-attend the
    /// candidate pool and predict object/no-object plus per-field assignments.
    ///
    /// * `query_states` – `(Q, H)` schema query embeddings
    /// * `candidate_states` – `(C, H)` from `candidate_encoder` in `score_sample`
    /// * `candidate_spans` – `(C, 2)` start/end indices
    /// * `candidate_valid` – `(C,)` validity mask
    /// * `field_query_ids` – which query index each field uses
    ///
    /// Returns `(object_logits[C], assign_logits[F][C+1])` where column 0 of
    /// each assign row is the null/ABSENT alternative.
    pub fn forward_group(
        &self,
        query_states: &[Vec<f32>],
        candidate_states: &[Vec<f32>],
        candidate_spans: &[(usize, usize)],
        candidate_valid: &[bool],
        field_query_ids: &[usize],
    ) -> Result<(
        Vec<f32>,                        // object_logits [I]
        Vec<Vec<Vec<f32>>>,              // assign_logits [I][F][1+C]
        Vec<(usize, usize)>,             // instance_spans [I]
    )> {
        let h = self.hidden_size;
        let d = self.record_dim;
        let c_count = candidate_states.len();
        let num_instances = Self::NUM_INSTANCE_QUERIES;
        let sqrt_d = (d as f32).sqrt();

        // Extract weights.
        let (w_inst, _, _) = mat2(self.inst_proj.weight())?;
        let b_inst = bias1(self.inst_proj.bias())?;
        let (w_field, _, _) = mat2(self.field_proj.weight())?;
        let b_field = bias1(self.field_proj.bias())?;
        let (w_cand, _, _) = mat2(self.cand_proj.weight())?;
        let b_cand = bias1(self.cand_proj.bias())?;
        let (w_q, _, _) = mat2(self.q_proj.weight())?;
        let b_q = bias1(self.q_proj.bias())?;
        let (w_k, _, _) = mat2(self.k_proj.weight())?;
        let b_k = bias1(self.k_proj.bias())?;
        let (w_v, _, _) = mat2(self.v_proj.weight())?;
        let b_v = bias1(self.v_proj.bias())?;
        let (w_obj, _, _) = mat2(self.object_head.weight())?;
        let b_obj = bias1(self.object_head.bias())?;
        let null_emb: Vec<f32> = self.null_embed.to_vec1().map_err(|e| GlinerError::inference(format!("{e}")))?;

        // Instance embeddings: [I, H]
        let inst_flat: Vec<Vec<f32>> = {
            let raw = self.instance_embed.to_vec2::<f32>().map_err(|e| GlinerError::inference(format!("{e}")))?;
            raw
        };

        // Cross-attend instance queries over candidate states.
        // q = q_proj(inst) [I, D], k = k_proj(cand) [C, D], v = v_proj(cand) [C, H]
        let inst_q: Vec<Vec<f32>> = inst_flat.iter().map(|x| apply_linear(&w_q, h, &b_q, x)).collect();
        let cand_k: Vec<Vec<f32>> = candidate_states.iter().map(|x| apply_linear(&w_k, h, &b_k, x)).collect();
        let cand_v: Vec<Vec<f32>> = candidate_states.iter().map(|x| apply_linear(&w_v, h, &b_v, x)).collect();

        let mask_logit = MASK_LOGIT;
        let mut inst_states: Vec<Vec<f32>> = Vec::with_capacity(num_instances);
        for i in 0..num_instances {
            // attn[i][c] = dot(inst_q[i], cand_k[c]) / sqrt_d
            let mut attn: Vec<f32> = (0..c_count)
                .map(|c| {
                    if candidate_valid[c] {
                        dot(&inst_q[i], &cand_k[c]) / sqrt_d
                    } else {
                        mask_logit
                    }
                })
                .collect();
            // softmax
            let max_attn = attn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = attn.iter().map(|v| (v - max_attn).exp()).sum();
            for v in attn.iter_mut() {
                *v = (*v - max_attn).exp() / exp_sum;
            }
            // pooled = sum(attn[c] * cand_v[c])
            let mut pooled = vec![0.0f32; h];
            for c in 0..c_count {
                for k in 0..h {
                    pooled[k] += attn[c] * cand_v[c][k];
                }
            }
            // instance = inst + pooled
            let mut state = vec![0.0f32; h];
            for k in 0..h {
                state[k] = inst_flat[i][k] + pooled[k];
            }
            inst_states.push(state);
        }

        // Object logits: object_head(inst_states) → [I]
        let object_logits: Vec<f32> = inst_states
            .iter()
            .map(|s| {
                let mut acc = b_obj[0];
                for k in 0..h {
                    acc += w_obj[k] * s[k];
                }
                acc
            })
            .collect();

        // Instance spans: use the candidate with highest object logit that is
        // a valid candidate. For anchorless mode, instances don't directly
        // correspond to pool candidates, so we assign spans via the latent
        // seed head scored against candidate states.
        let (w_lat, _, _) = mat2(self.latent_seed_head.weight())?;
        let b_lat = bias1(self.latent_seed_head.bias())?;
        let cand_scores_for_inst: Vec<Vec<f32>> = candidate_states
            .iter()
            .map(|cs| {
                let mut acc = b_lat[0];
                for k in 0..h {
                    acc += w_lat[k] * cs[k];
                }
                vec![acc]
            })
            .collect();
        // For each instance, pick the best valid candidate as its "anchor span".
        let instance_spans: Vec<(usize, usize)> = inst_states
            .iter()
            .map(|_| {
                // Find the valid candidate with highest latent_seed score.
                let mut best_c = 0;
                let mut best_s = f32::NEG_INFINITY;
                for c in 0..c_count {
                    if candidate_valid[c] && cand_scores_for_inst[c][0] > best_s {
                        best_s = cand_scores_for_inst[c][0];
                        best_c = c;
                    }
                }
                candidate_spans.get(best_c).copied().unwrap_or((0, 0))
            })
            .collect();

        // Per-field assignment logits: [I][F][1+C]
        let field_q_embs: Vec<Vec<f32>> = field_query_ids
            .iter()
            .map(|&qid| {
                if qid < query_states.len() {
                    query_states[qid].clone()
                } else {
                    vec![0.0f32; h]
                }
            })
            .collect();
        let field_q_proj: Vec<Vec<f32>> = field_q_embs
            .iter()
            .map(|x| apply_linear(&w_field, h, &b_field, x))
            .collect();

        let mut assign_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_instances);
        for i in 0..num_instances {
            let inst_q_proj = apply_linear(&w_inst, h, &b_inst, &inst_states[i]);
            let mut field_logits = Vec::with_capacity(field_query_ids.len());
            for f in 0..field_query_ids.len() {
                // query = inst_proj(inst) + field_proj(field) → [D]
                let mut query = vec![0.0f32; d];
                for k in 0..d {
                    query[k] = inst_q_proj[k] + field_q_proj[f][k];
                }
                // null column: dot(query, null_embed)
                let null_col = dot(&query, &null_emb);
                // candidate columns: dot(query, cand_proj(cand[c])) for each c
                let mut row = Vec::with_capacity(1 + c_count);
                row.push(null_col);
                for c in 0..c_count {
                    let cand_p = apply_linear(&w_cand, d, &b_cand, &candidate_states[c]);
                    let sc = if candidate_valid[c] {
                        dot(&query, &cand_p)
                    } else {
                        mask_logit
                    };
                    row.push(sc);
                }
                field_logits.push(row);
            }
            assign_logits.push(field_logits);
        }

        Ok((object_logits, assign_logits, instance_spans))
    }

    /// Decode one record group into a list of `DecodedRecord`.
    ///
    /// Uses the anchorless mode: sigmoid(object_logits) → object probability,
    /// then per-field softmax/sigmoid → field assignment.
    pub fn decode_group(
        object_logits: &[f32],
        assign_logits: &[Vec<Vec<f32>>],
        instance_spans: &[(usize, usize)],
        candidate_spans: &[(usize, usize)],
        candidate_valid: &[bool],
        field_query_ids: &[usize],
        object_threshold: f32,
        field_threshold: f32,
    ) -> Vec<DecodedRecord> {
        let num_instances = object_logits.len();
        let num_fields = field_query_ids.len();
        if num_instances == 0 || num_fields == 0 {
            return Vec::new();
        }

        // Select instances by object probability.
        let obj_prob: Vec<f32> = object_logits.iter().map(|&x| sigmoid_f32(x)).collect();
        let mut order: Vec<usize> = (0..num_instances).collect();
        order.sort_by(|&a, &b| obj_prob[b].partial_cmp(&obj_prob[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let selected: Vec<usize> = order
            .into_iter()
            .filter(|&i| obj_prob[i] >= object_threshold)
            .collect();

        let mut records = Vec::new();
        for &inst in &selected {
            let mut rec = DecodedRecord {
                fields: std::collections::HashMap::new(),
                field_scores: std::collections::HashMap::new(),
                score: obj_prob[inst],
            };

            for f in 0..num_fields {
                let qid = field_query_ids[f];
                let logits = &assign_logits[inst][f];
                let c_count = logits.len() - 1;
                if c_count == 0 {
                    continue;
                }

                // softmax over [null, cand1, ..., candC]
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|v| (v - max_logit).exp()).sum();
                let probs: Vec<f32> = logits.iter().map(|v| (v - max_logit).exp() / exp_sum).collect();

                // Pick the highest-probability column.
                let best_col = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if best_col == 0 {
                    // Null selected; field is absent.
                    continue;
                }
                let cand_idx = best_col - 1;
                if !candidate_valid.get(cand_idx).copied().unwrap_or(false) {
                    continue;
                }
                if probs[best_col] < field_threshold {
                    continue;
                }
                let span = candidate_spans[cand_idx];
                rec.fields.entry(qid).or_default().push(span);
                rec.field_scores.entry(qid).or_default().push(probs[best_col]);
            }

            if !rec.fields.is_empty() {
                records.push(rec);
            }
        }

        // Deduplicate anchorless records by field assignments.
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut deduped: Vec<DecodedRecord> = Vec::new();
        for rec in records {
            let key: Vec<(usize, Vec<(usize, usize)>)> = {
                let mut k: Vec<_> = rec.fields.iter().map(|(&qid, spans)| (qid, spans.clone())).collect();
                k.sort_by_key(|x| x.0);
                k
            };
            let key_val = format!("{key:?}");
            if let Some(existing) = seen.get(&key_val) {
                if rec.score > deduped[*existing].score {
                    deduped[*existing] = rec;
                }
            } else {
                let idx = deduped.len();
                seen.insert(key_val, idx);
                deduped.push(rec);
            }
        }

        deduped
    }
}

/// Complete GLiNER2.5 boundary model (everything except the shared encoder).
#[derive(Debug)]
pub struct BoundaryModel {
    /// Boundary-state encoder.
    pub encoder: BoundaryEncoder,
    /// Per-query start/end/inside projections.
    pub query_head: BoundaryQueryHead,
    /// Sparse proposal stage.
    pub proposer: BoundaryProposer,
    /// Shared candidate pool builder.
    pub pool_builder: SharedPoolBuilder,
    /// Shared candidate pool scorer.
    pub pool_scorer: SharedPoolScorer,
    /// Pairwise reranker.
    pub pair_scorer: PairScorer,
    /// Count prediction head.
    pub count_head: Option<Linear>,
    /// Abstention ("null") projection.
    pub null_projection: Option<Linear>,
    /// Candidate-content encoder.
    pub candidate_encoder: Option<Linear>,
    /// Classification head.
    pub classifier: BoundaryClassifier,
    /// Joint-IE relation scorer.
    pub relation_scorer: Option<RelationScorer>,
    /// Structured-record decoder.
    pub record_decoder: Option<RecordDecoder>,
    /// Boundary-head configuration.
    pub config: BoundaryConfig,
    /// Hidden size of the shared encoder.
    pub hidden_size: usize,
}

impl BoundaryModel {
    /// Load the full boundary model from a root VarBuilder.
    ///
    /// Every submodule is constructed eagerly so shape mismatches surface at
    /// load time rather than at first inference.
    pub fn load(vb: VarBuilder, config: &ExtractorConfig, device: &Device) -> Result<Self> {
        if config.architecture != Architecture::Gliner25 {
            return Err(GlinerError::model_loading(
                "BoundaryModel::load called for a non-boundary architecture",
            ));
        }
        let bh = vb.pp("boundary_head");
        let cfg = config.boundary.clone();
        let hidden_size = config.hidden_size;

        let optional_linear =
            |root: &VarBuilder, name: &str, enabled: bool| -> Result<Option<Linear>> {
                if !enabled {
                    return Ok(None);
                }
                match root.pp(name).get((hidden_size,), "bias") {
                    Ok(_) => Ok(Some(linear(hidden_size, 1, root.pp(name))?)),
                    Err(_) => Ok(None),
                }
            };

        Ok(Self {
            encoder: BoundaryEncoder::load(bh.pp("boundary_encoder"), hidden_size, &cfg, device)?,
            query_head: BoundaryQueryHead::load(bh.pp("boundary_query_head"), hidden_size, &cfg)?,
            proposer: BoundaryProposer::load(
                bh.pp("boundary_proposer"),
                hidden_size,
                cfg.boundary_dim,
            )?,
            pool_builder: SharedPoolBuilder::load(bh.pp("shared_pool_builder"), cfg.boundary_dim)?,
            pool_scorer: SharedPoolScorer::load(bh.pp("shared_pool_scorer"), hidden_size, &cfg)?,
            pair_scorer: PairScorer::load(bh.pp("pair_scorer"), hidden_size, &cfg)?,
            count_head: optional_linear(&bh, "count_head", cfg.enable_count_head)?,
            null_projection: optional_linear(&bh, "null_projection", true)?,
            candidate_encoder: match vb.get(
                (hidden_size, 2 * cfg.boundary_dim),
                "candidate_encoder.weight",
            ) {
                Ok(_) => Some(linear(
                    2 * cfg.boundary_dim,
                    hidden_size,
                    vb.pp("candidate_encoder"),
                )?),
                Err(_) => None,
            },
            classifier: BoundaryClassifier::load(vb.pp("classifier"), hidden_size)?,
            relation_scorer: if cfg.enable_relations {
                Some(RelationScorer::load(vb.pp("relation_scorer"), hidden_size)?)
            } else {
                None
            },
            record_decoder: if cfg.enable_records {
                Some(RecordDecoder::load(vb.pp("record_decoder"), hidden_size, device)?)
            } else {
                None
            },
            config: cfg,
            hidden_size,
        })
    }

    /// Read-only access to the classification head.
    pub fn classifier_ref(&self) -> Option<&BoundaryClassifier> {
        Some(&self.classifier)
    }

    /// Score candidate spans for a single sample via the shared-pool path.
    ///
    /// Mirrors upstream `DocumentCandidatePool` + `SharedPoolScorer`
    /// (`gliner2/models/boundary/pool.py`): union-marginal top-k pairing,
    /// per-query quota, key dedup, then FiLM-conditioned scoring of every
    /// (candidate, query) pair with marginal and inside-evidence corrections.
    ///
    /// * `text_states`: `(L, hidden_size)` word-level encoder states
    /// * `text_len`: number of valid words (<= L)
    /// * `query_states`: `(Q, hidden_size)` query embeddings
    ///
    /// Returns candidate spans plus row-major scores `[C][Q]`.
    pub fn score_sample(
        &self,
        text_states: &Tensor,
        text_len: usize,
        query_states: &Tensor,
    ) -> Result<SharedPoolScores> {
        let h = self.hidden_size;
        let d = self.config.boundary_dim;
        let c_dim = self.config.content_dim;
        let l = text_states.dims()[0];
        let q = query_states.dims()[0];
        let n = l + 1;
        let sqrt_d = (d as f32).sqrt();

        // --- Boundary states -------------------------------------------------
        let bs = self
            .encoder
            .forward(&text_states.unsqueeze(0)?, &[text_len])?
            .squeeze(0)?;
        let bs_rows = bs
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        let queries_flat = query_states
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        let text_flat = text_states
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;

        let boundary_valid = |i: usize| i <= text_len;

        // --- Weight extraction -------------------------------------------------
        // Query head: start/end/inside projections.
        let (w_sbp, _, _) = mat2(&self.query_head.start_boundary_projection.weight())?;
        let b_sbp = bias1(self.query_head.start_boundary_projection.bias())?;
        let (w_ebp, _, _) = mat2(&self.query_head.end_boundary_projection.weight())?;
        let b_ebp = bias1(self.query_head.end_boundary_projection.bias())?;
        let (w_sq, _, _) = mat2(&self.query_head.start_query_projection.weight())?;
        let b_sq = bias1(self.query_head.start_query_projection.bias())?;
        let (w_eq, _, _) = mat2(&self.query_head.end_query_projection.weight())?;
        let b_eq = bias1(self.query_head.end_query_projection.bias())?;
        let (w_it, _, _) = mat2(&self.query_head.inside_text_projection.weight())?;
        let b_it = bias1(self.query_head.inside_text_projection.bias())?;
        let (w_iq, _, _) = mat2(&self.query_head.inside_query_projection.weight())?;
        let b_iq = bias1(self.query_head.inside_query_projection.bias())?;

        // Projected keys/queries.
        let sk: Vec<Vec<f32>> = bs_rows
            .iter()
            .map(|x| apply_linear(&w_sbp, d, &b_sbp, x))
            .collect();
        let ek: Vec<Vec<f32>> = bs_rows
            .iter()
            .map(|x| apply_linear(&w_ebp, d, &b_ebp, x))
            .collect();
        let sq: Vec<Vec<f32>> = queries_flat
            .iter()
            .map(|x| apply_linear(&w_sq, h, &b_sq, x))
            .collect();
        let eq: Vec<Vec<f32>> = queries_flat
            .iter()
            .map(|x| apply_linear(&w_eq, h, &b_eq, x))
            .collect();
        let itk: Vec<Vec<f32>> = text_flat
            .iter()
            .map(|x| apply_linear(&w_it, h, &b_it, x))
            .collect();
        let iq: Vec<Vec<f32>> = queries_flat
            .iter()
            .map(|x| apply_linear(&w_iq, h, &b_iq, x))
            .collect();

        // --- Marginal logits -------------------------------------------------
        let mut start_logits = vec![vec![MASK_LOGIT; n]; q];
        let mut end_logits = vec![vec![MASK_LOGIT; n]; q];
        for qi in 0..q {
            for i in 0..n {
                if boundary_valid(i) {
                    start_logits[qi][i] = dot(&sq[qi], &sk[i]) / sqrt_d;
                    end_logits[qi][i] = dot(&eq[qi], &ek[i]) / sqrt_d;
                }
            }
        }

        // Inside logits over tokens plus per-query prefix sums over boundaries:
        // prefix[0] = 0; prefix[i+1] = sum of centered inside logits [0..i].
        let mut inside_mean = vec![0.0f32; q];
        let mut inside_prefix = vec![vec![0.0f32; n + 1]; q];
        for qi in 0..q {
            let mut logits = vec![0.0f32; l];
            let mut sum = 0.0f32;
            for t in 0..text_len.min(l) {
                logits[t] = dot(&iq[qi], &itk[t]) / sqrt_d;
                sum += logits[t];
            }
            inside_mean[qi] = sum / (text_len.max(1) as f32);
            let mut acc = 0.0f32;
            for i in 0..=n {
                inside_prefix[qi][i] = acc;
                if i < l && i < text_len {
                    acc += logits[i] - inside_mean[qi];
                }
            }
        }

        // --- DocumentCandidatePool -------------------------------------------
        const POOL_BOUNDARY_TOP_K: usize = 32;
        let pool_size = self.config.pool_size;
        const MIN_POOL_PER_QUERY: usize = 8;

        let mut union_start = vec![MASK_LOGIT; n];
        let mut union_end = vec![MASK_LOGIT; n];
        for i in 0..n {
            if !boundary_valid(i) {
                continue;
            }
            for qi in 0..q {
                union_start[i] = union_start[i].max(start_logits[qi][i]);
                union_end[i] = union_end[i].max(end_logits[qi][i]);
            }
        }

        let top_starts: Vec<usize> = argsort_desc_stable(&union_start)
            .into_iter()
            .filter(|&i| boundary_valid(i))
            .take(POOL_BOUNDARY_TOP_K)
            .collect();
        let top_ends: Vec<usize> = argsort_desc_stable(&union_end)
            .into_iter()
            .filter(|&i| boundary_valid(i))
            .take(POOL_BOUNDARY_TOP_K)
            .collect();

        let mut pair_s: Vec<usize> = Vec::new();
        let mut pair_e: Vec<usize> = Vec::new();
        let mut pair_valid: Vec<bool> = Vec::new();
        for &s in &top_starts {
            for &e in &top_ends {
                pair_s.push(s);
                pair_e.push(e);
                pair_valid.push(e > s && boundary_valid(s) && boundary_valid(e));
            }
        }

        let (w_pbs, _, _) = mat2(&self.pool_builder.start_projection.weight())?;
        let b_pbs = bias1(self.pool_builder.start_projection.bias())?;
        let (w_pbe, _, _) = mat2(&self.pool_builder.end_projection.weight())?;
        let b_pbe = bias1(self.pool_builder.end_projection.bias())?;
        let pbsk: Vec<Vec<f32>> = bs_rows
            .iter()
            .map(|x| apply_linear(&w_pbs, d, &b_pbs, x))
            .collect();
        let pbek: Vec<Vec<f32>> = bs_rows
            .iter()
            .map(|x| apply_linear(&w_pbe, d, &b_pbe, x))
            .collect();

        let compat: Vec<f32> = (0..pair_s.len())
            .map(|p| dot(&pbsk[pair_s[p]], &pbek[pair_e[p]]) / sqrt_d)
            .collect();
        let union_pair_score: Vec<f32> = (0..pair_s.len())
            .map(|p| compat[p] + union_start[pair_s[p]] + union_end[pair_e[p]])
            .collect();

        // Per-query quota reservations (priority band above global scores).
        let quota = MIN_POOL_PER_QUERY.min(pair_s.len());
        let mut all_keys: Vec<(usize, usize)> = Vec::new();
        let mut all_scores: Vec<f32> = Vec::new();
        let mut all_valid: Vec<bool> = Vec::new();
        if quota > 0 && q > 0 {
            for qi in 0..q {
                let pq: Vec<f32> = (0..pair_s.len())
                    .map(|p| {
                        if !pair_valid[p] {
                            MASK_LOGIT
                        } else {
                            start_logits[qi][pair_s[p]]
                                + end_logits[qi][pair_e[p]]
                                + compat[p]
                        }
                    })
                    .collect();
                let order = argsort_desc_stable(&pq);
                for (rank, &p) in order.iter().take(quota).enumerate() {
                    all_keys.push((pair_s[p], pair_e[p]));
                    all_scores.push(-MASK_LOGIT * 0.5 + (quota - rank) as f32);
                    all_valid.push(pair_valid[p]);
                }
            }
        }
        for p in 0..pair_s.len() {
            all_keys.push((pair_s[p], pair_e[p]));
            all_scores.push(union_pair_score[p]);
            all_valid.push(pair_valid[p]);
        }

        let (sel_s, sel_e, sel_valid) =
            dedup_pool(&all_keys, &all_scores, &all_valid, pool_size, n);
        let c_count = sel_s.len();

        // Retained candidates' compat prior.
        let selected_compat: Vec<f32> = (0..c_count)
            .map(|i| {
                if sel_valid[i] {
                    dot(&pbsk[sel_s[i]], &pbek[sel_e[i]]) / sqrt_d
                } else {
                    0.0
                }
            })
            .collect();

        // --- SharedPoolScorer --------------------------------------------------
        let (w_ss, _, _) = mat2(&self.pool_scorer.start_projection.weight())?;
        let b_ss = bias1(self.pool_scorer.start_projection.bias())?;
        let (w_se, _, _) = mat2(&self.pool_scorer.end_projection.weight())?;
        let b_se = bias1(self.pool_scorer.end_projection.bias())?;
        let (w_len, _, _) = mat2(&self.pool_scorer.length_projection.weight())?;
        let b_len = bias1(self.pool_scorer.length_projection.bias())?;
        let (w_prior, _, _) = mat2(&self.pool_scorer.prior_projection.weight())?;
        let b_prior = bias1(self.pool_scorer.prior_projection.bias())?;
        let (w_cproj, _, _) = mat2(&self.pool_scorer.content_projection.weight())?;
        let b_cproj = bias1(self.pool_scorer.content_projection.bias())?;
        let (w_qproj, _, _) = mat2(&self.pool_scorer.query_projection.weight())?;
        let b_qproj = bias1(self.pool_scorer.query_projection.bias())?;
        let (w_film, _, _) = mat2(&self.pool_scorer.film.weight())?;
        let b_film = bias1(self.pool_scorer.film.bias())?;
        let (w_fo_h, _, _) = mat2(&self.pool_scorer.film_output_hidden.weight())?;
        let b_fo_h = bias1(self.pool_scorer.film_output_hidden.bias())?;
        let (w_fo_o, _, _) = mat2(&self.pool_scorer.film_output_out.weight())?;
        let b_fo_o = bias1(self.pool_scorer.film_output_out.bias())?;
        let pool_norm = self.pool_scorer.candidate_norm_params()?;

        // Span-content pooling: value-project tokens, mean over the span via a
        // running-sum, then LayerNorm (content_soft_max_pool = false).
        let (w_cv, _, _) = mat2(&self.pair_scorer.content_value_weight())?;
        let b_cv = bias1(self.pair_scorer.content_value_bias().as_ref())?;
        let (ln_w, ln_b) = self.pair_scorer.content_ln_params()?;
        let token_values: Vec<Vec<f32>> = text_flat
            .iter()
            .map(|x| apply_linear(&w_cv, h, &b_cv, x))
            .collect();
        // Running sums per content dimension over valid tokens.
        let mut run_sum = vec![vec![0.0f32; c_dim]; n + 1];
        for t in 0..l {
            for k in 0..c_dim {
                run_sum[t + 1][k] =
                    run_sum[t][k] + if t < text_len { token_values[t][k] } else { 0.0 };
            }
        }

        // Candidate composition.
        let tl = text_len.max(1) as f32;
        let mut candidates: Vec<Vec<f32>> = Vec::with_capacity(c_count);
        for i in 0..c_count {
            if !sel_valid[i] {
                candidates.push(vec![0.0; d]);
                continue;
            }
            let (s, e) = (sel_s[i], sel_e[i]);
            let len_f = (e - s).max(1) as f32;
            let feats = [(1.0f32 + len_f).ln(), len_f / tl, 1.0 / len_f.sqrt()];
            let span_sum: Vec<f32> = (0..c_dim)
                .map(|k| run_sum[e.min(n)][k] - run_sum[s.min(n)][k])
                .collect();
            let pooled_content_raw: Vec<f32> =
                span_sum.iter().map(|v| v / len_f).collect();
            let pooled_content = layernorm(&pooled_content_raw, &ln_w, &ln_b)
                .unwrap_or_else(|e| panic!("content layernorm: {e}"));

            let mut cand = vec![0.0f32; d];
            for ri in 0..d {
                let mut acc = b_ss[ri] + b_se[ri] + b_len[ri] + b_prior[ri] + b_cproj[ri] * 0.0;
                let wsr = &w_ss[ri * d..(ri + 1) * d];
                let wer = &w_se[ri * d..(ri + 1) * d];
                let wlr = &w_len[ri * 3..(ri + 1) * 3];
                let wpr = &w_prior[ri];
                let wcr = &w_cproj[ri * c_dim..(ri + 1) * c_dim];
                for k in 0..d {
                    acc += wsr[k] * bs_rows[s][k] + wer[k] * bs_rows[e][k];
                }
                for k in 0..3 {
                    acc += wlr[k] * feats[k];
                }
                acc += wpr * selected_compat[i];
                for k in 0..c_dim {
                    acc += wcr[k] * pooled_content[k];
                }
                cand[ri] = acc;
            }
                {
                    let (nw, nb) = (&pool_norm.0, &pool_norm.1);
                    cand = layernorm(&cand, nw, nb)
                        .unwrap_or_else(|e| panic!("candidate norm: {e}"));
                }
                candidates.push(cand);
        }

        // Query projection + FiLM conditioning.
        let qproj: Vec<Vec<f32>> = queries_flat
            .iter()
            .map(|x| apply_linear(&w_qproj, h, &b_qproj, x))
            .collect();
        let films: Vec<(Vec<f32>, Vec<f32>)> = qproj
            .iter()
            .map(|qp| {
                let fb = apply_linear(&w_film, d, &b_film, qp);
                (fb[..d].to_vec(), fb[d..].to_vec())
            })
            .collect();

        let mut scores = vec![vec![MASK_LOGIT; q]; c_count];
        for ci in 0..c_count {
            if !sel_valid[ci] {
                continue;
            }
            let (s, e) = (sel_s[ci], sel_e[ci]);
            let len_i = (e - s).max(1);
            for qi in 0..q {
                let mut sc = dot(&candidates[ci], &qproj[qi]) / sqrt_d;
                let (gamma, beta) = (&films[qi].0, &films[qi].1);
                let conditioned: Vec<f32> = (0..d)
                    .map(|k| candidates[ci][k] * (1.0 + gamma[k]) + beta[k])
                    .collect();
                let fh = apply_linear(&w_fo_h, d, &b_fo_h, &conditioned);
                let fh_act: Vec<f32> = fh
                    .iter()
                    .map(|v| v * 0.5 * (1.0 + erf(v / std::f32::consts::SQRT_2)))
                    .collect();
                let fo = apply_linear(&w_fo_o, c_dim, &b_fo_o, &fh_act);
                sc += fo[0];
                sc += start_logits[qi][s] + end_logits[qi][e];
                let interval = inside_prefix[qi][(e).min(n)] - inside_prefix[qi][(s).min(n)]
                    + inside_mean[qi] * len_i as f32;
                sc += interval / (len_i as f32).sqrt();
                scores[ci][qi] = sc;
            }
        }

        // --- Candidate states for record decoder ----------------------------
        let cand_states = if let Some(ref ce) = self.candidate_encoder {
            let (w_ce, _, _) = mat2(ce.weight())?;
            let b_ce = bias1(ce.bias())?;
            let input_dim = 2 * d;
            Some(
                (0..c_count)
                    .map(|i| {
                        if !sel_valid[i] {
                            return vec![0.0f32; h];
                        }
                        let mut inp = Vec::with_capacity(input_dim);
                        inp.extend_from_slice(&bs_rows[sel_s[i]]);
                        inp.extend_from_slice(&bs_rows[sel_e[i]]);
                        apply_linear(&w_ce, input_dim, &b_ce, &inp)
                    })
                    .collect(),
            )
        } else {
            None
        };

                Ok(SharedPoolScores {
            starts: sel_s,
            ends: sel_e,
            valid: sel_valid,
            scores,
            candidate_states: cand_states,
            boundary_states: Some(bs_rows),
        })
    }
}

/// Result of `BoundaryModel::score_sample`.
#[derive(Debug, Clone)]
pub struct SharedPoolScores {
    /// Candidate start boundaries (token index).
    pub starts: Vec<usize>,
    /// Candidate end boundaries (exclusive token index).
    pub ends: Vec<usize>,
    /// Whether each candidate slot is real.
    pub valid: Vec<bool>,
    /// Row-major `[C][Q]` reranked logits.
    pub scores: Vec<Vec<f32>>,
    /// H-dimensional candidate states `[C][H]` (from `candidate_encoder`).
    /// `None` when the model lacks a `candidate_encoder` (no record decoder).
    pub candidate_states: Option<Vec<Vec<f32>>>,
    /// Boundary encoder states `[L+1][d]` for relation scoring.
    /// `None` when the model lacks a `relation_scorer`.
    pub boundary_states: Option<Vec<Vec<f32>>>,
}

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu_f32(x: f32) -> f32 {
    x * 0.5 * (1.0 + ((2.0_f32).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 approximation (|err| < 1.5e-7)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - t * (0.254829592
            - t * (0.284496736
                - t * (1.421413741 - t * (1.453152027 - t * 1.061405429))));
    sign * y
}

fn layernorm(x: &[f32], weight: &[f32], bias: &[f32]) -> Result<Vec<f32>> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (var + BOUNDARY_LAYER_NORM_EPS as f32).sqrt();
    Ok(x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((&v, &w), &b)| (v - mean) * inv * w + b)
        .collect())
}

fn dedup_pool(
    keys: &[(usize, usize)],
    scores: &[f32],
    valid: &[bool],
    capacity: usize,
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
    let invalid_key = (n, n);
    // Sort score desc stable (invalid slots sink).
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| {
        let sa = if valid[a] { scores[a] } else { MASK_LOGIT };
        let sb = if valid[b] { scores[b] } else { MASK_LOGIT };
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    // Sort by key asc stable among that order.
    let mut by_key = order.clone();
    by_key.sort_by_key(|&i| {
        if valid[i] {
            keys[i]
        } else {
            invalid_key
        }
    });
    // Keep first occurrence of each key among valid entries.
    let mut seen = std::collections::HashSet::new();
    let mut keep_order: Vec<usize> = Vec::new();
    for &i in &by_key {
        if !valid[i] {
            continue;
        }
        if seen.insert(keys[i]) {
            keep_order.push(i);
        }
    }
    // Sort survivors by score desc and truncate.
    keep_order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    keep_order.truncate(capacity.max(1));
    let starts = keep_order.iter().map(|&i| keys[i].0).collect();
    let ends = keep_order.iter().map(|&i| keys[i].1).collect();
    let kept_valid = vec![true; keep_order.len()];
    (starts, ends, kept_valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu() -> Device { Device::Cpu }

    // ── mat2 ────────────────────────────────────────────────────────────

    #[test]
    fn mat2_2d() {
        let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &cpu()).unwrap();
        let (v, rows, cols) = mat2(&t).unwrap();
        assert_eq!(rows, 3);
        assert_eq!(cols, 2);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn mat2_1d() {
        let t = Tensor::new(&[10.0f32, 20.0, 30.0], &cpu()).unwrap();
        let (v, rows, cols) = mat2(&t).unwrap();
        assert_eq!(rows, 3);
        assert_eq!(cols, 1);
        assert_eq!(v, vec![10.0, 20.0, 30.0]);
    }

    // ── apply_linear ────────────────────────────────────────────────────

    #[test]
    fn apply_linear_identity() {
        // y = I·x + 0
        let w = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity, row-major
        let b = vec![0.0, 0.0];
        let x = vec![3.0, 4.0];
        let y = apply_linear(&w, 2, &b, &x);
        assert_eq!(y.len(), 2);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn apply_linear_with_bias() {
        // w = [[1, 2], [3, 4]], b = [10, 20], x = [1, 1]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0];
        let x = vec![1.0, 1.0];
        let y = apply_linear(&w, 2, &b, &x);
        assert!((y[0] - 13.0).abs() < 1e-6); // 1*1+2*1+10
        assert!((y[1] - 27.0).abs() < 1e-6); // 3*1+4*1+20
    }

    #[test]
    fn apply_linear_projection() {
        // 3×2 weight: maps 2-dim input to 3-dim output
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 5.0];
        let x = vec![2.0, 3.0];
        let y = apply_linear(&w, 2, &b, &x);
        assert_eq!(y.len(), 3);
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 3.0).abs() < 1e-6);
        assert!((y[2] - 10.0).abs() < 1e-6); // 2+3+5
    }

    // ── dot ──────────────────────────────────────────────────────────────

    #[test]
    fn dot_basic() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_zero() {
        assert!((dot(&[1.0, 2.0], &[0.0, 0.0])).abs() < 1e-6);
    }

    #[test]
    fn dot_single() {
        assert!((dot(&[7.0], &[3.0]) - 21.0).abs() < 1e-6);
    }

    // ── argsort_desc_stable ──────────────────────────────────────────────

    #[test]
    fn argsort_desc_stable_basic() {
        let vals = [1.0, 3.0, 2.0, 3.0];
        let idx = argsort_desc_stable(&vals);
        // 3.0 at index 1 comes before 3.0 at index 3 (stable by index)
        assert_eq!(idx[0], 1);
        assert_eq!(idx[1], 3);
        assert_eq!(idx[2], 2);
        assert_eq!(idx[3], 0);
    }

    #[test]
    fn argsort_desc_stable_empty() {
        let idx = argsort_desc_stable(&[]);
        assert!(idx.is_empty());
    }

    #[test]
    fn argsort_desc_stable_single() {
        let idx = argsort_desc_stable(&[42.0]);
        assert_eq!(idx, vec![0]);
    }

    // ── bias1 ────────────────────────────────────────────────────────────

    #[test]
    fn bias1_some() {
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &cpu()).unwrap();
        let b = bias1(Some(&t)).unwrap();
        assert_eq!(b, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn bias1_none() {
        let b = bias1(None).unwrap();
        assert!(b.is_empty());
    }

    // ── ln_params ────────────────────────────────────────────────────────

    #[test]
    fn ln_params_with_bias() {
        let w = Tensor::new(&[1.0f32, 2.0], &cpu()).unwrap();
        let b = Tensor::new(&[0.5f32, 0.6], &cpu()).unwrap();
        let mut hm = std::collections::HashMap::new();
        hm.insert("weight".to_string(), w.clone());
        hm.insert("bias".to_string(), b.clone());
        let vb = candle_nn::VarBuilder::from_tensors(hm, candle_core::DType::F32, &cpu());
        let ln = candle_nn::layer_norm(2, 1e-5, vb).unwrap();
        let (weights, biases) = ln_params(&ln).unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 2.0).abs() < 1e-6);
        assert_eq!(biases.len(), 2);
        assert!((biases[0] - 0.5).abs() < 1e-6);
        assert!((biases[1] - 0.6).abs() < 1e-6);
    }

    // ── dedup_pool ───────────────────────────────────────────────────────

    #[test]
    fn dedup_pool_no_duplicates() {
        let keys = vec![(0, 2), (1, 3), (2, 4)];
        let scores = vec![3.0, 1.0, 2.0];
        let valid = vec![true, true, true];
        let (s, e, v) = dedup_pool(&keys, &scores, &valid, 10, 5);
        assert_eq!(s.len(), 3);
        assert!(v.iter().all(|&x| x));
        // Sorted by score desc: (0,2)=3.0, (2,4)=2.0, (1,3)=1.0
        assert_eq!(s, vec![0, 2, 1]);
        assert_eq!(e, vec![2, 4, 3]);
    }

    #[test]
    fn dedup_pool_with_duplicates() {
        // Same key (1,2) appears twice; only the higher-scored one kept
        let keys = vec![(1, 2), (1, 2), (3, 4)];
        let scores = vec![5.0, 8.0, 1.0];
        let valid = vec![true, true, true];
        let (s, e, _) = dedup_pool(&keys, &scores, &valid, 10, 5);
        assert_eq!(s.len(), 2); // two unique keys
        // Higher score for (1,2) was 8.0, so order: (1,2)=8.0, (3,4)=1.0
        assert_eq!(s[0], 1);
        assert_eq!(e[0], 2);
        assert_eq!(s[1], 3);
        assert_eq!(e[1], 4);
    }

    #[test]
    fn dedup_pool_invalid_entries() {
        let keys = vec![(0, 2), (1, 3)];
        let scores = vec![5.0, 3.0];
        let valid = vec![false, true]; // first is invalid
        let (s, e, v) = dedup_pool(&keys, &scores, &valid, 10, 5);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], 1);
        assert_eq!(e[0], 3);
        assert!(v[0]);
    }

    #[test]
    fn dedup_pool_truncation() {
        let keys: Vec<(usize, usize)> = (0..20).map(|i| (i, i + 1)).collect();
        let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let valid = vec![true; 20];
        let (s, _, _) = dedup_pool(&keys, &scores, &valid, 5, 25);
        assert_eq!(s.len(), 5);
    }

    // ── BoundaryConfig ──────────────────────────────────────────────────

    #[test]
    fn boundary_config_defaults() {
        let cfg = BoundaryConfig::default();
        assert_eq!(cfg.boundary_dim, 128);
        assert_eq!(cfg.content_dim, 64);
        assert_eq!(cfg.pair_dim, 128);
        assert_eq!(cfg.pool_size, 192);
        assert!(cfg.enable_relations);
        assert!(cfg.enable_records);
    }

    #[test]
    fn boundary_config_from_hf_json() {
        let json = r#"{
            "boundary_head": {
                "boundary_dim": 256,
                "content_dim": 128,
                "enable_relations": false,
                "pool_size": 64
            }
        }"#;
        let cfg = BoundaryConfig::from_hf_config_json(json);
        assert_eq!(cfg.boundary_dim, 256);
        assert_eq!(cfg.content_dim, 128);
        assert_eq!(cfg.pool_size, 64);
        assert!(!cfg.enable_relations);
        // Unspecified fields keep defaults
        assert_eq!(cfg.pair_dim, 128);
        assert!(cfg.enable_records);
    }

    #[test]
    fn boundary_config_from_empty_json() {
        let cfg = BoundaryConfig::from_hf_config_json("{}");
        let def = BoundaryConfig::default();
        assert_eq!(cfg.boundary_dim, def.boundary_dim);
        assert_eq!(cfg.pool_size, def.pool_size);
    }

    // ── BoundaryEncoder shape test (random weights) ─────────────────────

    #[test]
    fn encoder_forward_shapes() {
        let device = cpu();
        let hidden = 16;
        let bd = 8;
        let text_len = 5;
        let n = text_len + 1; // boundaries

        let text = Tensor::randn(0f32, 1.0, (1, text_len, hidden), &device).unwrap();
        assert_eq!(text.dims(), &[1, text_len, hidden]);
        let bs_dummy = Tensor::randn(0f32, 1.0, (1, n, bd), &device).unwrap();
        assert_eq!(bs_dummy.dims(), &[1, n, bd]);
    }

    // ── score_sample math verification ──────────────────────────────────

    #[test]
    fn inside_prefix_sum_basic() {
        // Verify the prefix sum logic used inside score_sample
        let logits = vec![1.0f32, 2.0, 3.0];
        let mean = 2.0f32;
        let n = 4; // 3 tokens + 1 boundary
        let l = 3;
        let mut prefix = vec![0.0f32; n + 1];
        let mut acc = 0.0f32;
        for i in 0..=n {
            prefix[i] = acc;
            if i < l {
                acc += logits[i] - mean;
            }
        }
        // prefix[0]=0, prefix[1]=1-2=-1, prefix[2]=-1+2-2=-1, prefix[3]=-1+3-2=0
        assert!((prefix[0]).abs() < 1e-6);
        assert!((prefix[1] - (-1.0)).abs() < 1e-6);
        assert!((prefix[2] - (-1.0)).abs() < 1e-6);
        assert!((prefix[3]).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_of_negative_scores() {
        // Verify sigmoid produces values below typical thresholds for negative scores
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        assert!(sigmoid(-1.52) < 0.3);
        assert!(sigmoid(-1.52) > 0.1);
        assert!(sigmoid(0.0) == 0.5);
        assert!(sigmoid(2.0) > 0.8);
    }

    // ── RecordDecoder decode_group ─────────────────────────────────────

    #[test]
    fn decode_group_no_instances() {
        let records = RecordDecoder::decode_group(&[], &[], &[], &[], &[], &[], 0.5, 0.5);
        assert!(records.is_empty());
    }

    #[test]
    fn decode_group_selects_above_threshold() {
        // 2 instances, 1 field, 2 candidates
        let object_logits = vec![-2.0, 3.0]; // sigmoid: ~0.12, ~0.95
        let assign_logits = vec![
            vec![vec![-1.0, 0.5, 1.0]],  // instance 0: null=0.27, c0=0.37, c1=0.37
            vec![vec![-3.0, 2.0, 5.0]],  // instance 1: null=0.01, c0=0.05, c1=0.94
        ];
        let instance_spans = vec![(0, 3), (1, 5)];
        let candidate_spans = vec![(0, 3), (1, 5)];
        let candidate_valid = vec![true, true];
        let field_query_ids = vec![0];

        let records = RecordDecoder::decode_group(
            &object_logits, &assign_logits, &instance_spans,
            &candidate_spans, &candidate_valid, &field_query_ids,
            0.5, 0.3,
        );
        // Only instance 1 (score ~0.95) should be selected.
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].score, sigmoid_f32(3.0));
        // Instance 1 picks candidate 1 (highest assign logit).
        assert_eq!(records[0].fields.get(&0), Some(&vec![(1, 5)]));
    }

    #[test]
    fn decode_group_null_field_skipped() {
        // Instance with high object score but field prefers null.
        let object_logits = vec![5.0];
        let assign_logits = vec![
            vec![vec![5.0, -1.0, -2.0]], // null=0.94, c0=0.01, c1=0.005
        ];
        let instance_spans = vec![(0, 3)];
        let candidate_spans = vec![(0, 3), (3, 6)];
        let candidate_valid = vec![true, true];
        let field_query_ids = vec![0];

        let records = RecordDecoder::decode_group(
            &object_logits, &assign_logits, &instance_spans,
            &candidate_spans, &candidate_valid, &field_query_ids,
            0.3, 0.3,
        );
        // Instance selected (score ~0.99), but field has no assignment (null chosen).
        assert!(records.is_empty());
    }

    #[test]
    fn decode_group_dedup() {
        // Two identical instances → should dedup to one record.
        let object_logits = vec![3.0, 3.0];
        let assign_logits = vec![
            vec![vec![-2.0, 5.0]],  // null=0.01, c0=0.99
            vec![vec![-2.0, 5.0]],  // same
        ];
        let instance_spans = vec![(0, 3), (0, 3)];
        let candidate_spans = vec![(0, 3)];
        let candidate_valid = vec![true];
        let field_query_ids = vec![0];

        let records = RecordDecoder::decode_group(
            &object_logits, &assign_logits, &instance_spans,
            &candidate_spans, &candidate_valid, &field_query_ids,
            0.3, 0.3,
        );
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn gelu_f32_basic() {
        assert!((gelu_f32(0.0) - 0.0).abs() < 1e-6);
        // GELU is monotonic and near-identity for large positives
        assert!(gelu_f32(10.0) > 9.0);
        // GELU(1) > 0, GELU(-1) < 0
        assert!(gelu_f32(1.0) > 0.0);
        assert!(gelu_f32(-1.0) < 0.0);
    }

    fn zero_linear(in_dim: usize, out_dim: usize) -> Linear {
        Linear::new(
            Tensor::zeros((out_dim, in_dim), DType::F32, &cpu()).unwrap(),
            Some(Tensor::zeros(out_dim, DType::F32, &cpu()).unwrap()),
        )
    }

    #[test]
    fn relation_scorer_forward_empty() {
        let d = 8;
        let h = 8;
        let rs = RelationScorer {
            head_content_projection: zero_linear(h, h),
            tail_content_projection: zero_linear(h, h),
            relation_content_gate: zero_linear(2 * h, h),
            mlp_hidden: zero_linear(6 * h + 2, h),
            mlp_out: zero_linear(h, 1),
            content_linear: zero_linear(4 * h, 1),
        };
        let bs = vec![vec![0.0f32; d]; 5]; // L+1=5
        let rel_q = vec![0.0f32; 2 * h];
        let result = rs.forward(&bs, &rel_q, &[], &[], 4).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn relation_scorer_forward_shapes() {
        let d = 8;
        let h = 8;
        let rs = RelationScorer {
            head_content_projection: zero_linear(h, h),
            tail_content_projection: zero_linear(h, h),
            relation_content_gate: zero_linear(2 * h, h),
            mlp_hidden: zero_linear(6 * h + 2, h),
            mlp_out: zero_linear(h, 1),
            content_linear: zero_linear(4 * h, 1),
        };
        let bs = vec![vec![0.1f32; d]; 10];
        let rel_q = vec![0.2f32; 2 * h];
        let heads = vec![(0, 3), (1, 4)];
        let tails = vec![(5, 8), (6, 9)];
        let result = rs.forward(&bs, &rel_q, &heads, &tails, 8).unwrap();
        assert_eq!(result.len(), 2);
        // With zero weights, score should be 0 for all pairs
        for &s in &result {
            assert!(s.abs() < 1e-5, "expected ~0, got {s}");
        }
    }
}
