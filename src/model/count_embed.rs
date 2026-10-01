// Rust guideline compliant 2026-04-03
//! Count embedding layer for GLiNER2.
//!
//! Two checkpoint variants exist upstream, both keyed off the same
//! `pos_embedding` + `gru` trunk and differing only in what consumes the GRU
//! output:
//!
//! | variant | upstream | tail | tensors |
//! |---------|----------|------|---------|
//! | [`CountLayer::Lstm`] | `layers.py::CountLSTM` | concat + 2-layer MLP | 9 |
//! | [`CountLayer::V2`] | `layers.py::CountLSTMv2` | `DownscaledTransformer` | 37 |
//!
//! Which one a checkpoint ships is decided by its weight keys, not by
//! `counting_layer` in `config.json`: `gliner2-base-v1` and
//! `GLiNER2.5-Decide` both declare `count_lstm`, but only the former carries
//! the transformer block.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

/// Count embedding layer output.
#[derive(Debug, Clone)]
pub struct CountEmbedOutput {
    /// Count-aware entity embeddings: (pred_count, num_entity_types, hidden)
    pub embeddings: Tensor,
    /// Predicted count value
    pub pred_count: usize,
}

/// The count positions a checkpoint supports.
pub const MAX_COUNT: usize = 20;

/// Multi-head attention for the transformer.
struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let in_proj_weight = vb.get((3 * embed_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * embed_dim, "in_proj_bias")?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim: embed_dim / num_heads,
            embed_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // batch_first=True shape: (batch, seq, embed)
        let (batch_size, seq_len, _) = x.dims3()?;

        // Q, K, V projection
        let x_2d = x.reshape((batch_size * seq_len, self.embed_dim))?;
        let qkv = x_2d.matmul(&self.in_proj_weight.t()?)?;
        let qkv = qkv.broadcast_add(&self.in_proj_bias)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3 * self.embed_dim))?;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, self.embed_dim)?;
        let k = qkv.narrow(2, self.embed_dim, self.embed_dim)?;
        let v = qkv.narrow(2, 2 * self.embed_dim, self.embed_dim)?;

        // Reshape for multi-head attention: (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores
        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)?)?;
        let attn_scores = (attn_scores / scale)?;

        let attn_probs = if let Some(m) = mask {
            let masked = attn_scores.add(m)?;
            candle_nn::ops::softmax(&masked, 3)?
        } else {
            candle_nn::ops::softmax(&attn_scores, 3)?
        };

        // Context: (batch, seq, embed)
        let context = attn_probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.embed_dim,
        ))?;

        self.out_proj.forward(&context)
    }
}

/// Transformer encoder layer.
struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    fn load(vb: VarBuilder, d_model: usize, nhead: usize, dim_feedforward: usize) -> Result<Self> {
        let self_attn = MultiHeadAttention::load(vb.pp("self_attn"), d_model, nhead)?;
        let linear1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("linear2"))?;
        let norm1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let attn_out = self.self_attn.forward(x, None)?;
        let x = x.add(&attn_out)?;
        let x = self.norm1.forward(&x)?;

        // Feed-forward with residual (TransformerEncoderLayer default activation is ReLU)
        let ff_out = self.linear1.forward(&x)?;
        let ff_out = ff_out.relu()?;
        let ff_out = self.linear2.forward(&ff_out)?;
        let x = x.add(&ff_out)?;
        self.norm2.forward(&x)
    }
}

/// GRU layer implementation.
struct GRULayer {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    hidden_size: usize,
}

impl GRULayer {
    fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        let weight_ih = vb.get((3 * hidden_size, input_size), "weight_ih_l0")?;
        let weight_hh = vb.get((3 * hidden_size, hidden_size), "weight_hh_l0")?;
        let bias_ih = vb.get(3 * hidden_size, "bias_ih_l0")?;
        let bias_hh = vb.get(3 * hidden_size, "bias_hh_l0")?;

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, h_0: Option<&Tensor>) -> Result<Tensor> {
        let (seq_len, batch_size, _) = x.dims3()?;

        let mut h = if let Some(h) = h_0 {
            h.clone()
        } else {
            Tensor::zeros((batch_size, self.hidden_size), DType::F32, x.device())?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.narrow(0, t, 1)?.squeeze(0)?;

            // Gate computations (PyTorch GRU-compatible):
            // r = sigmoid(x_r + h_r + b_ir + b_hr)
            // z = sigmoid(x_z + h_z + b_iz + b_hz)
            // n = tanh(x_n + r * h_n + b_in + b_hn)
            // h' = (1 - z) * n + z * h
            let x_proj = x_t.matmul(&self.weight_ih.t()?)?;
            let h_proj = h.matmul(&self.weight_hh.t()?)?;

            let chunk_size = self.hidden_size;

            let x_r = x_proj.narrow(1, 0, chunk_size)?;
            let x_z = x_proj.narrow(1, chunk_size, chunk_size)?;
            let x_n = x_proj.narrow(1, 2 * chunk_size, chunk_size)?;

            let h_r = h_proj.narrow(1, 0, chunk_size)?;
            let h_z = h_proj.narrow(1, chunk_size, chunk_size)?;
            let h_n = h_proj.narrow(1, 2 * chunk_size, chunk_size)?;

            let b_ir = self.bias_ih.narrow(0, 0, chunk_size)?;
            let b_iz = self.bias_ih.narrow(0, chunk_size, chunk_size)?;
            let b_in = self.bias_ih.narrow(0, 2 * chunk_size, chunk_size)?;

            let b_hr = self.bias_hh.narrow(0, 0, chunk_size)?;
            let b_hz = self.bias_hh.narrow(0, chunk_size, chunk_size)?;
            let b_hn = self.bias_hh.narrow(0, 2 * chunk_size, chunk_size)?;

            let r_pre = x_r.add(&h_r)?.broadcast_add(&b_ir.add(&b_hr)?)?;
            let z_pre = x_z.add(&h_z)?.broadcast_add(&b_iz.add(&b_hz)?)?;

            // Sigmoid: 1 / (1 + exp(-x))
            let neg_r = r_pre.neg()?;
            let exp_neg_r = neg_r.exp()?;
            let ones_r = Tensor::full(1.0f32, exp_neg_r.dims(), exp_neg_r.device())?;
            let r_t = exp_neg_r.add(&ones_r)?.recip()?;

            let neg_z = z_pre.neg()?;
            let exp_neg_z = neg_z.exp()?;
            let ones_z = Tensor::full(1.0f32, exp_neg_z.dims(), exp_neg_z.device())?;
            let z_t = exp_neg_z.add(&ones_z)?.recip()?;

            let n_pre = x_n.add(&r_t.mul(&h_n)?)?.broadcast_add(&b_in.add(&b_hn)?)?;
            let n_t = n_pre.tanh()?;

            let one_minus_z = z_t.ones_like()?.sub(&z_t)?;
            let h_new = one_minus_z.mul(&n_t)?.add(&z_t.mul(&h)?)?;

            outputs.push(h_new.unsqueeze(0)?);
            h = h_new;
        }

        Tensor::cat(&outputs, 0)
    }
}

/// The `pos_embedding` + `gru` trunk both upstream variants share.
///
/// Returns the GRU output alongside the broadcast query embeddings, because
/// `CountLSTM` concatenates them while `CountLSTMv2` adds them.
struct GruTrunk {
    pos_embedding: Embedding,
    gru: GRULayer,
    hidden_size: usize,
    device: Device,
}

impl GruTrunk {
    fn load(vb: VarBuilder, hidden_size: usize, device: Device) -> Result<Self> {
        Ok(Self {
            pos_embedding: candle_nn::embedding(MAX_COUNT, hidden_size, vb.pp("pos_embedding"))?,
            gru: GRULayer::load(vb.pp("gru"), hidden_size, hidden_size)?,
            hidden_size,
            device,
        })
    }

    /// Run the count-step sequence: positional embeddings in, per-step query
    /// embeddings out.
    fn run(&self, entity_embs: &Tensor, pred_count: usize) -> Result<(Tensor, Tensor)> {
        let num_types = entity_embs.dims()[0];
        let count = pred_count.min(MAX_COUNT);

        let pos_ids: Vec<u32> = (0..count).map(|i| i as u32).collect();
        let pos_embs =
            self.pos_embedding
                .forward(&Tensor::from_slice(&pos_ids, (count,), &self.device)?)?;

        let query = entity_embs
            .unsqueeze(0)?
            .broadcast_as((count, num_types, self.hidden_size))?;
        let gru_out = self.gru.forward(
            &pos_embs
                .unsqueeze(1)?
                .broadcast_as((count, num_types, self.hidden_size))?,
            Some(entity_embs),
        )?;

        Ok((gru_out, query))
    }
}

/// `layers.py::CountLSTM`: `projector(concat([gru_out, query]))`.
///
/// Ships as `count_embed.{pos_embedding, gru, projector.{0,2}}` — 9 tensors,
/// with no transformer block.
pub struct CountLstmLayer {
    trunk: GruTrunk,
    projector_0: Linear,
    projector_2: Linear,
}

impl CountLstmLayer {
    fn from_var_builder(vb: VarBuilder, hidden_size: usize, device: Device) -> Result<Self> {
        let trunk = GruTrunk::load(vb.clone(), hidden_size, device)?;
        // candle stores Linear weight as (out, in): projector.0 is 4*hidden
        // wide, projector.2 maps back down to hidden.
        let projector = vb.pp("projector");
        Ok(Self {
            trunk,
            projector_0: candle_nn::linear(2 * hidden_size, 4 * hidden_size, projector.pp("0"))?,
            projector_2: candle_nn::linear(4 * hidden_size, hidden_size, projector.pp("2"))?,
        })
    }

    fn forward(&self, entity_embs: &Tensor, pred_count: usize) -> Result<CountEmbedOutput> {
        let (gru_out, query) = self.trunk.run(entity_embs, pred_count)?;
        let concat = Tensor::cat(&[gru_out, query], 2)?;
        let out = self
            .projector_2
            .forward(&self.projector_0.forward(&concat)?.relu()?)?;
        Ok(CountEmbedOutput {
            embeddings: out,
            pred_count: pred_count.min(MAX_COUNT),
        })
    }
}

/// `layers.py::CountLSTMv2`: GRU + `DownscaledTransformer`.
pub struct CountEmbedLayer {
    trunk: GruTrunk,
    in_projector: Linear,
    transformer_layers: Vec<TransformerEncoderLayer>,
    out_projector_0: Linear,
    out_projector_2: Linear,
    out_projector_4: Linear,
}

impl CountEmbedLayer {
    /// Create a new randomly initialized count embedding layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Entity embedding size.
    /// * `device` - Target device for model parameters.
    ///
    /// # Returns
    ///
    /// A newly initialized count embedding layer.
    ///
    /// # Errors
    ///
    /// Returns an error if layer initialization fails.
    pub fn new(hidden_size: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::from_var_builder(vb, hidden_size, device)
    }

    /// Create a count embedding layer from VarBuilder with loaded weights.
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder rooted at the model weights.
    /// * `hidden_size` - Entity embedding size.
    /// * `max_count` - Maximum supported count positions.
    /// * `device` - Target device for model parameters.
    ///
    /// # Returns
    ///
    /// A count embedding layer initialized from existing weights.
    ///
    /// # Errors
    ///
    /// Returns an error if required weights are missing or invalid.
    pub fn from_var_builder(vb: VarBuilder, hidden_size: usize, device: Device) -> Result<Self> {
        // Load position embeddings + GRU (shared with CountLSTM)
        let trunk = GruTrunk::load(vb.clone(), hidden_size, device)?;

        // Load transformer components
        let in_projector =
            candle_nn::linear(hidden_size, 128, vb.pp("transformer").pp("in_projector"))?;

        let vb_transformer = vb.pp("transformer").pp("transformer");
        let mut transformer_layers = Vec::new();
        for i in 0..2 {
            let layer = TransformerEncoderLayer::load(
                vb_transformer.pp("layers").pp(i.to_string()),
                128,
                4,
                256,
            )?;
            transformer_layers.push(layer);
        }

        // Load output projector
        let out_projector_0 = candle_nn::linear(
            896,
            hidden_size,
            vb.pp("transformer").pp("out_projector").pp("0"),
        )?;
        let out_projector_2 = candle_nn::linear(
            hidden_size,
            hidden_size,
            vb.pp("transformer").pp("out_projector").pp("2"),
        )?;
        let out_projector_4 = candle_nn::linear(
            hidden_size,
            hidden_size,
            vb.pp("transformer").pp("out_projector").pp("4"),
        )?;

        Ok(Self {
            trunk,
            in_projector,
            transformer_layers,
            out_projector_0,
            out_projector_2,
            out_projector_4,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `entity_embs` - Entity embeddings, shape (num_entity_types, hidden)
    /// * `pred_count` - Predicted number of entities
    ///
    /// # Returns
    /// Count-aware entity embeddings, shape (pred_count, num_entity_types, hidden)
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, entity_embs: &Tensor, pred_count: usize) -> Result<CountEmbedOutput> {
        let (gru_out, query) = self.trunk.run(entity_embs, pred_count)?;

        // CountLSTMv2: x = transformer(gru_out + query), batch_first=True
        let transformer_input = gru_out.add(&query)?;
        let mut x = self.in_projector.forward(&transformer_input)?; // (count, num_types, 128)
        for layer in &self.transformer_layers {
            x = layer.forward(&x)?;
        }

        // DownscaledTransformer out projection:
        // concat([x, original_x], dim=-1) -> MLP(896 -> 768 -> 768 -> 768, ReLU)
        let concat = Tensor::cat(&[x, transformer_input], 2)?;
        let out = self.out_projector_0.forward(&concat)?;
        let out = out.relu()?;
        let out = self.out_projector_2.forward(&out)?;
        let out = out.relu()?;
        let out = self.out_projector_4.forward(&out)?;

        Ok(CountEmbedOutput {
            embeddings: out,
            pred_count: pred_count.min(MAX_COUNT),
        })
    }
}

/// Which count-aware projection a checkpoint ships.
///
/// Dispatching on an enum rather than on `config.counting_layer`: that field is
/// `count_lstm` for every span checkpoint, including the ones that carry the
/// `CountLSTMv2` transformer block. The weights are the only reliable signal,
/// which is why [`CountLayer::from_var_builder`] probes for both layouts.
pub enum CountLayer {
    /// `layers.py::CountLSTM` — GRU + concat + 2-layer MLP.
    Lstm(CountLstmLayer),
    /// `layers.py::CountLSTMv2` — GRU + `DownscaledTransformer`.
    V2(CountEmbedLayer),
}

impl CountLayer {
    /// A randomly initialized layer, for use before weights are loaded.
    ///
    /// `rebuild_model` replaces this once the checkpoint's real layout is known.
    /// `CountLSTM` is the smaller of the two, so it is the cheaper placeholder.
    pub fn placeholder(hidden_size: usize, device: Device) -> Self {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::Lstm(
            CountLstmLayer::from_var_builder(vb, hidden_size, device)
                .expect("placeholder CountLSTM shapes are always valid"),
        )
    }

    /// Build whichever variant the weights describe, or `Ok(None)` if the
    /// checkpoint ships no count-aware projection at all.
    ///
    /// The v2 probe checks `transformer.in_projector.weight`, which
    /// `CountLSTM` checkpoints do not have; the v1 probe checks
    /// `projector.0.weight`, which they do.
    pub fn from_var_builder(
        vb: VarBuilder,
        hidden_size: usize,
        device: Device,
    ) -> Result<Option<Self>> {
        const DOWNSCALED_DIM: usize = 128;

        // candle stores Linear weight as (out, in).
        if vb
            .pp("transformer")
            .pp("in_projector")
            .get((DOWNSCALED_DIM, hidden_size), "weight")
            .is_ok()
        {
            return CountEmbedLayer::from_var_builder(vb, hidden_size, device)
                .map(|layer| Some(Self::V2(layer)))
                .map_err(|e| candle_core::Error::Msg(format!("count_embed v2: {e}")));
        }

        if vb
            .pp("projector")
            .pp("0")
            .get((4 * hidden_size, 2 * hidden_size), "weight")
            .is_ok()
        {
            return CountLstmLayer::from_var_builder(vb, hidden_size, device)
                .map(|layer| Some(Self::Lstm(layer)))
                .map_err(|e| candle_core::Error::Msg(format!("count_embed lstm: {e}")));
        }

        Ok(None)
    }

    /// Transform query embeddings into count-aware embeddings.
    ///
    /// Returns `(pred_count, num_entity_types, hidden)`.
    pub fn forward(&self, entity_embs: &Tensor, pred_count: usize) -> Result<CountEmbedOutput> {
        match self {
            Self::Lstm(layer) => layer.forward(entity_embs, pred_count),
            Self::V2(layer) => layer.forward(entity_embs, pred_count),
        }
    }

    /// Which upstream class this layer corresponds to. Used in error messages.
    pub fn variant(&self) -> &'static str {
        match self {
            Self::Lstm(_) => "CountLSTM",
            Self::V2(_) => "CountLSTMv2",
        }
    }
}
