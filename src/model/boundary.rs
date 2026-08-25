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

use candle_core::{Device, Tensor};
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
}

/// Structured-record decoder (`record_decoder.*` at the root).
#[derive(Debug)]
pub struct RecordDecoder {
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

impl RecordDecoder {
    /// Number of learned record-instance queries.
    pub const NUM_INSTANCE_QUERIES: usize = 32;

    /// Load from `vb` rooted at `record_decoder`.
    pub fn load(vb: VarBuilder, hidden_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            cand_proj: linear(hidden_size, 128, vb.pp("cand_proj"))?,
            field_proj: linear(hidden_size, 128, vb.pp("field_proj"))?,
            inst_proj: linear(hidden_size, 128, vb.pp("inst_proj"))?,
            k_proj: linear(hidden_size, 128, vb.pp("k_proj"))?,
            q_proj: linear(hidden_size, 128, vb.pp("q_proj"))?,
            v_proj: linear(hidden_size, hidden_size, vb.pp("v_proj"))?,
            instance_embed: vb
                .get((Self::NUM_INSTANCE_QUERIES, hidden_size), "instance_embed")?
                .to_device(device)?,
            null_embed: vb.get((128,), "null_embed")?.to_device(device)?,
            latent_seed_head: linear(hidden_size, 1, vb.pp("latent_seed_head"))?,
            object_head: linear(hidden_size, 1, vb.pp("object_head"))?,
        })
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
            .map(|x| {
                let v = apply_linear(&w_cv, h, &b_cv, x);
                layernorm(&v, &ln_w, &ln_b).unwrap_or_else(|e| panic!("content layernorm: {e}"))
            })
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
            let pooled_content: Vec<f32> =
                span_sum.iter().map(|v| v / len_f).collect();

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
            let _ = &pool_norm; // extracted below
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
                    .map(|v| v * 0.5 * (1.0 + (v * SQRT_2_OVER_PI).tanh()))
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

        Ok(SharedPoolScores {
            starts: sel_s,
            ends: sel_e,
            valid: sel_valid,
            scores,
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
}

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

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
