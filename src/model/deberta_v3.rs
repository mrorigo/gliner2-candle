// Rust guideline compliant 2026-04-03
//! DeBERTa V3 encoder for GLiNER2.
//!
//! This implements the exact DeBERTa V3 architecture used by GLiNER2:
//! - Disentangled multi-head attention with relative position bias (c2p + p2c)
//! - query_proj, key_proj, value_proj naming
//! - Relative position embeddings (rel_embeddings)
//! - No token_type_embeddings

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct DebertaV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub max_relative_positions: isize,
    pub pos_att_type: Vec<String>,
    pub position_buckets: usize,
    pub share_att_key: bool,
    pub relative_attention: bool,
}

impl Default for DebertaV3Config {
    fn default() -> Self {
        Self {
            vocab_size: 128011,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            position_buckets: 256,
            share_att_key: true,
            relative_attention: true,
        }
    }
}

/// DeBERTa V3 embeddings (word_embeddings + LayerNorm, no token_type_embeddings)
struct DebertaV3Embeddings {
    word_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl DebertaV3Embeddings {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.word_embeddings.forward(input_ids)?;
        self.layer_norm.forward(&embeddings)
    }
}

/// Self-attention with disentangled relative position bias (DeBERTa V3 style)
struct DebertaV3Attention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_dense: Linear,
    output_layer_norm: LayerNorm,
    num_attention_heads: usize,
    attention_head_size: usize,
    max_relative_positions: isize,
    position_buckets: usize,
    share_att_key: bool,
    pos_att_type: Vec<String>,
}

impl DebertaV3Attention {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let vb_self = vb.pp("attention").pp("self");
        let vb_out = vb.pp("attention").pp("output");
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let query_proj = candle_nn::linear(
            config.hidden_size,
            config.hidden_size,
            vb_self.pp("query_proj"),
        )?;
        let key_proj = candle_nn::linear(
            config.hidden_size,
            config.hidden_size,
            vb_self.pp("key_proj"),
        )?;
        let value_proj = candle_nn::linear(
            config.hidden_size,
            config.hidden_size,
            vb_self.pp("value_proj"),
        )?;

        let output_dense =
            candle_nn::linear(config.hidden_size, config.hidden_size, vb_out.pp("dense"))?;
        let output_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb_out.pp("LayerNorm"),
        )?;

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_dense,
            output_layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            max_relative_positions: config.max_relative_positions,
            position_buckets: config.position_buckets,
            share_att_key: config.share_att_key,
            pos_att_type: config.pos_att_type.clone(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        rel_embeddings: Option<&Tensor>,
        rel_idx: Option<&RelPositionIndex>,
    ) -> Result<Tensor> {
        let input_tensor = hidden_states.clone();
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

        // Project Q, K, V
        let query_states = self.query_proj.forward(hidden_states)?;
        let key_states = self.key_proj.forward(hidden_states)?;
        let value_states = self.value_proj.forward(hidden_states)?;

        // Reshape for multi-head attention: (batch, seq, heads, head_size) -> (batch, heads, seq, head_size)
        let query_layer = query_states
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_layer = key_states
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_layer = value_states
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        let prof = std::env::var("GLINER2_PROFILE").is_ok();
        let t_qkv = std::time::Instant::now();
        // Compute scale factor based on pos_att_type
        let mut scale_factor = 1.0f64;
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1.0;
        }
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1.0;
        }
        let scale = (self.attention_head_size as f64 * scale_factor).sqrt();

        // Content-based attention scores: (batch, heads, seq, head_size) @ (batch, heads, head_size, seq)
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        attention_scores = (attention_scores / scale)?;

        if prof { eprintln!("PROFILE     qkv+qk={:?} seq={seq_len}", t_qkv.elapsed()); }
        // Add disentangled attention bias if relative embeddings are available
        let t_rel = std::time::Instant::now();
        match (rel_embeddings, rel_idx) {
            (Some(rel_emb), Some(idx)) => {
                let rel_att = self.disentangled_attention_bias(
                    &query_layer,
                    &key_layer,
                    rel_emb,
                    idx,
                )?;
                attention_scores = attention_scores.add(&rel_att)?;
            }
            (Some(rel_emb), None) => {
                let idx = RelPositionIndex::build(
                    batch_size,
                    seq_len,
                    self.num_attention_heads,
                    rel_emb.dims()[0] / 2,
                    query_layer.device(),
                )?;
                let rel_att =
                    self.disentangled_attention_bias(&query_layer, &key_layer, rel_emb, &idx)?;
                attention_scores = attention_scores.add(&rel_att)?;
            }
            _ => {}
        }
        if prof { eprintln!("PROFILE     rel_bias={:?}", t_rel.elapsed()); }

        // Apply attention mask
        let t_sm = std::time::Instant::now();
        attention_scores = attention_scores.add(attention_mask)?;
        let attention_probs = candle_nn::ops::softmax(&attention_scores, 3)?;
        if prof { eprintln!("PROFILE     mask+softmax={:?}", t_sm.elapsed()); }

        // Context layer: (batch, heads, seq, seq) @ (batch, heads, seq, head_size)
        let context = attention_probs.matmul(&value_layer)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.reshape((batch_size, seq_len, hidden_size))?;

        // Output projection + residual + layer norm
        let output = self.output_dense.forward(&context)?;
        let output = output.add(&input_tensor)?;
        self.output_layer_norm.forward(&output)
    }

    fn scale(&self) -> f64 {
        let mut scale_factor = 1.0f64;
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1.0;
        }
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1.0;
        }
        (self.attention_head_size as f64 * scale_factor).sqrt()
    }

    /// Compute disentangled attention bias with c2p and p2c components.
    ///
    /// `idx` holds pre-broadcast U32 gather indices shared across all layers.
    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        rel_embeddings: &Tensor,
        idx: &RelPositionIndex,
    ) -> Result<Tensor> {
        let inv_scale = 1.0 / self.scale();
        let batch_size = query_layer.dims()[0];
        let mut score: Option<Tensor> = None;

        // Content-to-Position (c2p): out[i, j] = q[i] . pk[idx[i, j]] / scale
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            let pos_key = if self.share_att_key {
                self.key_proj.forward(rel_embeddings)?
            } else {
                candle_core::bail!("share_att_key=false not implemented")
            };
            let pk = (self.pos_proj(pos_key, batch_size)? * inv_scale)?;
            let att = query_layer.matmul(&pk.transpose(2, 3)?)?;
            let part = gather_along_last_dim(&att, &idx.idx)?;
            score = Some(match score {
                Some(s) => s.add(&part)?,
                None => part,
            });
        }

        // Position-to-Content (p2c): out[i, j] = k[j] . pq[idx[i, j]] / scale
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            let pos_query = if self.share_att_key {
                self.query_proj.forward(rel_embeddings)?
            } else {
                candle_core::bail!("share_att_key=false not implemented")
            };
            let pq = (self.pos_proj(pos_query, batch_size)? * inv_scale)?;
            let att = key_layer.matmul(&pq.transpose(2, 3)?)?;
            let part = gather_last_dim_transposed(&att, &idx.idx)?;
            score = Some(match score {
                Some(s) => s.add(&part)?,
                None => part,
            });
        }

        match score {
            Some(s) => Ok(s),
            None => candle_core::bail!("pos_att_type must contain c2p or p2c"),
        }
    }

    /// Project position embeddings to per-head layout (batch, heads, head_size, P).
    fn pos_proj(&self, proj: Tensor, batch_size: usize) -> Result<Tensor> {
        let p = proj.dims()[0];
        Ok(proj
            .reshape((p, self.num_attention_heads, self.attention_head_size))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .broadcast_as((
                batch_size,
                self.num_attention_heads,
                p,
                self.attention_head_size,
            ))?
            .contiguous()?)
    }
}

/// Precomputed relative-position gather indices shared across all layers.
///
/// `idx[i, j] = clamp(i - j + span)` in `[0, 2*span)`, shape
/// `(batch, heads, seq, seq)`, dtype U32, contiguous.
///
/// This single tensor serves both attention directions:
/// - c2p: `out[i, j] = q[i] . pk[idx[i, j]]` (direct last-dim gather)
/// - p2c: `out[i, j] = k[j] . pq[idx[i, j]]` (gather + transpose fused)
pub(crate) struct RelPositionIndex {
    pub idx: Tensor,
}

impl RelPositionIndex {
    fn build(
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        att_span: usize,
        device: &Device,
    ) -> Result<Self> {
        let q_ids = Tensor::arange(0i64, seq_len as i64, device)?.unsqueeze(1)?;
        let k_ids = Tensor::arange(0i64, seq_len as i64, device)?.unsqueeze(0)?;
        let rel_pos_f = q_ids.broadcast_sub(&k_ids)?.to_dtype(DType::F32)?;

        let span_t = Tensor::full(att_span as f32, (seq_len, seq_len), device)?;
        let max_val = (att_span * 2 - 1) as f32;
        let dims = (batch_size, num_heads, seq_len, seq_len);

        let idx = rel_pos_f
            .add(&span_t)?
            .clamp(0.0, max_val)?
            .to_dtype(DType::U32)?
            .broadcast_as(dims)?
            .contiguous()?;

        Ok(Self { idx })
    }
}

/// Feed-forward layer
struct DebertaV3Intermediate {
    dense: Linear,
    output_dense: Linear,
    output_layer_norm: LayerNorm,
}

impl DebertaV3Intermediate {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let vb_int = vb.pp("intermediate");
        let vb_out = vb.pp("output");
        let dense = candle_nn::linear(
            config.hidden_size,
            config.intermediate_size,
            vb_int.pp("dense"),
        )?;
        let output_dense = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb_out.pp("dense"),
        )?;
        let output_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb_out.pp("LayerNorm"),
        )?;

        Ok(Self {
            dense,
            output_dense,
            output_layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_tensor = hidden_states.clone();
        let hidden = self.dense.forward(hidden_states)?;
        let hidden = hidden.gelu()?;
        let output = self.output_dense.forward(&hidden)?;
        let output = output.add(&input_tensor)?;
        self.output_layer_norm.forward(&output)
    }
}

/// DeBERTa V3 layer
struct DebertaV3Layer {
    attention: DebertaV3Attention,
    intermediate: DebertaV3Intermediate,
}

impl DebertaV3Layer {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let attention = DebertaV3Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV3Intermediate::load(vb, config)?;
        Ok(Self {
            attention,
            intermediate,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        rel_embeddings: Option<&Tensor>,
        rel_idx: Option<&RelPositionIndex>,
    ) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let hidden = self.attention.forward(
            hidden_states,
            attention_mask,
            rel_embeddings,
            rel_idx,
        )?;
        let t_attn = t0.elapsed();
        let t1 = std::time::Instant::now();
        let out = self.intermediate.forward(&hidden);
        if std::env::var("GLINER2_PROFILE").is_ok() {
            eprintln!("PROFILE     attn={t_attn:?} ffn={:?}", t1.elapsed());
        }
        out
    }
}

/// DeBERTa V3 encoder
struct DebertaV3Encoder {
    layers: Vec<DebertaV3Layer>,
    num_attention_heads: usize,
}

impl DebertaV3Encoder {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DebertaV3Layer::load(vb.pp("layer").pp(i.to_string()), config)?;
            layers.push(layer);
        }
        Ok(Self {
            layers,
            num_attention_heads: config.num_attention_heads,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let profile = std::env::var("GLINER2_PROFILE").is_ok();
        let t_all = std::time::Instant::now();
        let mut hidden = hidden_states.clone();

        // Relative-position gather indices depend only on shapes: build once
        // and share across all layers instead of rebuilding per layer.
        let rel_idx = match rel_embeddings {
            Some(rel) => {
                let (batch_size, seq_len, _) = hidden_states.dims3()?;
                Some(RelPositionIndex::build(
                    batch_size,
                    seq_len,
                    self.num_attention_heads,
                    rel.dims()[0] / 2,
                    hidden_states.device(),
                )?)
            }
            None => None,
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if profile && (i == 0 || i == self.layers.len() - 1) {
                let t0 = std::time::Instant::now();
                hidden = layer.forward(&hidden, attention_mask, rel_embeddings, rel_idx.as_ref())?;
                let el = t0.elapsed();
                eprintln!("PROFILE   layer {i}: {el:?} seq={}", hidden.dims()[1]);
            } else {
                hidden = layer.forward(&hidden, attention_mask, rel_embeddings, rel_idx.as_ref())?;
            }
        }
        if profile {
            eprintln!("PROFILE   encoder_total={:?} seq={}", t_all.elapsed(), hidden.dims()[1]);
        }
        Ok(hidden)
    }
}

/// DeBERTa V3 model (matches GLiNER2's encoder architecture)
pub struct DebertaV3Model {
    embeddings: DebertaV3Embeddings,
    encoder: DebertaV3Encoder,
    rel_embeddings: Option<Embedding>,
    rel_layer_norm: Option<LayerNorm>,
    device: Device,
}

impl DebertaV3Model {
    /// Load a DeBERTa V3 model from a `VarBuilder`.
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder rooted at encoder weights.
    /// * `config` - DeBERTa model configuration.
    ///
    /// # Returns
    ///
    /// A loaded DeBERTa V3 model.
    ///
    /// # Errors
    ///
    /// Returns an error if required weights are missing or invalid.
    pub fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let embeddings = DebertaV3Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV3Encoder::load(vb.pp("encoder"), config)?;

        // Load relative position embeddings from encoder.rel_embeddings
        let rel_embeddings = if vb.contains_tensor("encoder.rel_embeddings.weight") {
            Some(candle_nn::embedding(
                512,
                config.hidden_size,
                vb.pp("encoder").pp("rel_embeddings"),
            )?)
        } else {
            None
        };
        let rel_layer_norm = if vb.contains_tensor("encoder.LayerNorm.weight") {
            Some(candle_nn::layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("encoder").pp("LayerNorm"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            rel_embeddings,
            rel_layer_norm,
            device: vb.device().clone(),
        })
    }

    /// Run a forward pass through the DeBERTa V3 encoder.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs tensor.
    /// * `token_type_ids` - Token type IDs (unused for DeBERTa V3).
    /// * `attention_mask` - Optional attention mask.
    ///
    /// # Returns
    ///
    /// Encoder output embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;

        let attention_mask = match attention_mask {
            Some(mask) => mask.clone(),
            None => input_ids.ones_like()?,
        };

        // Create pairwise extended attention mask (matching HF DeBERTa logic)
        // Input: (batch, seq_len) with 1s for valid tokens, 0s for padding
        // Pairwise mask: (batch, 1, seq_len, seq_len), then broadcast to heads
        let attention_mask = match attention_mask.rank() {
            2 => {
                // (batch, seq) -> (batch, 1, 1, seq)
                let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
                // Pairwise validity: valid query AND valid key
                let pairwise_attention_mask = extended_attention_mask
                    .broadcast_mul(&extended_attention_mask.squeeze(2)?.unsqueeze(3)?)?;
                // Broadcast to (batch, heads, seq, seq)
                pairwise_attention_mask.broadcast_as((
                    pairwise_attention_mask.dims()[0],
                    self.encoder.num_attention_heads,
                    pairwise_attention_mask.dims()[2],
                    pairwise_attention_mask.dims()[3],
                ))?
            }
            3 => {
                let mask = attention_mask.unsqueeze(1)?;
                mask.broadcast_as((
                    mask.dims()[0],
                    self.encoder.num_attention_heads,
                    mask.dims()[2],
                    mask.dims()[3],
                ))?
            }
            _ => candle_core::bail!("Wrong shape for attention_mask"),
        };
        let attention_mask = attention_mask.to_dtype(DType::F32)?;
        // Convert binary mask to additive mask: 0 for valid, very negative for invalid
        let attention_mask = (attention_mask.ones_like()? - &attention_mask)?
            .broadcast_mul(&Tensor::try_from(f32::MIN)?.to_device(attention_mask.device())?)?;

        let rel_embeddings = self.rel_states();
        self.encoder.forward(
            &embedding_output,
            &attention_mask,
            rel_embeddings.as_ref(),
        )
    }

    fn rel_states(&self) -> Option<Tensor> {
        let raw = self.rel_embeddings.as_ref()?.embeddings();
        match &self.rel_layer_norm {
            Some(ln) => Some(ln.forward(raw).expect("rel LayerNorm forward")),
            None => Some(raw.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Debug forward returning [embedding_output, layer0_out, ..] for parity testing.
    pub fn forward_debug(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let mut stages = vec![embedding_output.clone()];

        let attention_mask = match attention_mask {
            Some(mask) => mask.clone(),
            None => input_ids.ones_like()?,
        };
        let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
        let pairwise_attention_mask = extended_attention_mask
            .broadcast_mul(&extended_attention_mask.squeeze(2)?.unsqueeze(3)?)?;
        let pairwise_attention_mask = pairwise_attention_mask.broadcast_as((
            pairwise_attention_mask.dims()[0],
            self.encoder.num_attention_heads,
            pairwise_attention_mask.dims()[2],
            pairwise_attention_mask.dims()[3],
        ))?;
        let attention_mask = pairwise_attention_mask.to_dtype(DType::F32)?;
        let attention_mask = (attention_mask.ones_like()? - &attention_mask)?
            .broadcast_mul(&Tensor::try_from(f32::MIN)?.to_device(attention_mask.device())?)?;

        let rel_embeddings = self.rel_states();
        let mut hidden = embedding_output;
        let rel_idx = match rel_embeddings.as_ref() {
            Some(rel) => {
                let (batch_size, seq_len, _) = hidden.dims3()?;
                Some(RelPositionIndex::build(
                    batch_size,
                    seq_len,
                    self.encoder.num_attention_heads,
                    rel.dims()[0] / 2,
                    input_ids.device(),
                )?)
            }
            None => None,
        };
        for layer in &self.encoder.layers {
            hidden = layer.forward(
                &hidden,
                &attention_mask,
                rel_embeddings.as_ref(),
                rel_idx.as_ref(),
            )?;
            stages.push(hidden.clone());
        }
        stages.push(hidden);
        Ok(stages)
    }
}

/// Gather along the last dimension with the output written transposed.
///
/// `input`: contiguous `(b, h, l, P)` f32.
/// `indices`: contiguous `(b, h, l, o)` U32 with values `< P`.
/// Result: `out[b, h, i, j] = input[b, h, j, indices[b, h, i, j]]`.
///
/// Fuses the gather and the `transpose(2, 3)` of the p2c path into a single
/// pass with contiguous output.
fn gather_last_dim_transposed(input: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let (b, h, l, p) = input.dims4()?;
    let o = indices.dims()[3];

    let src = input.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let idx = indices.to_dtype(DType::U32)?.contiguous()?.flatten_all()?.to_vec1::<u32>()?;

    let mut out = vec![0f32; b * h * l * o];
    for bh in 0..b * h {
        let base_in = bh * l;
        let base_out = base_in * o;
        for i in 0..l {
            let dst = &mut out[base_out + i * o..base_out + (i + 1) * o];
            let idx_row = &idx[base_out + i * o..base_out + (i + 1) * o];
            for (j, k) in idx_row.iter().enumerate() {
                dst[j] = src[(base_in + j) * p + *k as usize];
            }
        }
    }
    Tensor::from_vec(out, (b, h, l, o), input.device())
}

/// Gather values along the last dimension using per-row indices.
///
/// `input`: contiguous `(batch, heads, seq, P)` f32.
/// `indices`: contiguous `(batch, heads, seq, out)` U32 with values `< P`.
///
/// Direct row-copy kernel: candle's generic `gather` carries significant
/// per-element dispatch overhead on CPU; here each output element is a plain
/// indexed copy from a source row of length `P` (64), which runs near memory
/// bandwidth.
fn gather_along_last_dim(input: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let (b, h, l, p) = input.dims4()?;
    let o = indices.dims()[3];

    let src = input.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let idx = indices.to_dtype(DType::U32)?.contiguous()?.flatten_all()?.to_vec1::<u32>()?;

    let mut out = vec![0f32; b * h * l * o];
    for bh in 0..b * h {
        let base_in = bh * l;
        let base_out = base_in * o;
        for i in 0..l {
            let src_row = &src[(base_in + i) * p..(base_in + i + 1) * p];
            let dst = &mut out[base_out + i * o..base_out + (i + 1) * o];
            let idx_row = &idx[base_out + i * o..base_out + (i + 1) * o];
            for (j, k) in idx_row.iter().enumerate() {
                dst[j] = src_row[*k as usize];
            }
        }
    }
    Tensor::from_vec(out, (b, h, l, o), input.device())
}
