// Dorado SUP encoder and CRF beam search on Dorado's koi kernels.
//
// Port of dorado (commit 8b8fc5d) nn/ConvStack.cpp, nn/TxModules.cpp, nn/AuxiliaryData.cpp,
// basecall/CudaModelRunner.cpp and basecall/decode/CUDADecoder.cpp for the variable-chunk-size
// path Dorado uses on sm80+: same weight tiling, int8 quantisation, remove_bits rounding, fp16
// RoPE tables and koi call sequence. The fp8 path (Hopper only) is not ported; it is rejected at
// construction.

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {
#include "koi.h"
}

namespace {

struct KoiTensorExt : public KoiTensor {
    KoiTensorExt(const at::Tensor &t, const std::vector<int> &dim_tags) { init(t, dim_tags); }
    KoiTensorExt(const at::Tensor &t,
                 const std::vector<int> &dim_tags,
                 const at::Tensor &scale_t,
                 int scale_tag_) {
        init(t, dim_tags);
        if (scale_t.defined() && scale_t.numel()) {
            if (type_id != KOI_I8) {
                throw std::runtime_error("KoiTensor scale tensor only compatible with I8 tensor");
            }
            scale_tag = scale_tag_;
            scale_data = scale_t.data_ptr();
            scale_type_id = type_id_from_tensor(scale_t);
            if (scale_type_id != KOI_F32) {
                throw std::runtime_error("KoiTensor scale tensor must be F32");
            }
        }
    }

    static KoiTypeId type_id_from_tensor(const at::Tensor &t) {
        if (t.dtype() == torch::kF16) {
            return KOI_F16;
        } else if (t.dtype() == torch::kF32) {
            return KOI_F32;
        } else if (t.dtype() == torch::kFloat8_e4m3fn) {
            return KOI_E4M3;
        } else if (t.dtype() == torch::kI8) {
            return KOI_I8;
        }
        throw std::runtime_error("KoiTensor unsupported dtype");
    }

    void init(const at::Tensor &t, const std::vector<int> &dim_tags) {
        data_ptr = t.data_ptr();
        ndims = static_cast<int>(t.dim());
        type_id = type_id_from_tensor(t);
        if (ndims > KOI_TENSOR_MAX_DIMS || size_t(ndims) != dim_tags.size()) {
            throw std::runtime_error("KoiTensor dimension mismatch");
        }
        for (int i = 0; i < ndims; ++i) {
            dims[i].tag = dim_tags[i];
            dims[i].size = t.size(i);
            dims[i].stride = t.stride(i);
        }
        scale_data = nullptr;
        scale_tag = 0;
        scale_type_id = KOI_NONE;
    }
};

struct ScaledTensor {
    at::Tensor t;
    at::Tensor scale;
};

// dorado/torch_utils/tensor_utils.cpp
ScaledTensor quantize_tensor(const at::Tensor &t, int dim) {
    auto fp_range = t.abs().amax(dim);
    constexpr int levels = 256;
    auto quant_scale = ((levels / 2) / fp_range);
    auto quant_max = (levels / 2) - 1;
    auto t_quant = (t * quant_scale.unsqueeze(dim)).round().clip(-quant_max, quant_max);
    return ScaledTensor{t_quant.to(at::ScalarType::Char), quant_scale.to(at::ScalarType::Float)};
}

void apply_rounding(at::Tensor &t, int remove_bits) {
    if (t.dtype() == torch::kF16) {
        t.view(torch::kI16).add_(1 << (remove_bits - 1));
        t.view(torch::kI16).bitwise_and_(0x10000 - (1 << remove_bits));
    }
}

void check(int res, const char *what) {
    if (res != KOI_SUCCESS) {
        throw std::runtime_error(std::string("koi call failed: ") + what + " (" +
                                 std::to_string(res) + ")");
    }
}

struct Layer {
    ScaledTensor wqkv_i8, wqkv_f16, fc1_i8, fc1_f16;
    at::Tensor proj_weight, proj_bias, res_weights, res2_weights, fc2_wts;
    float alpha;
};

struct Conv {
    at::Tensor w_device, b_device;
    int C_in, C_out, winlen, stride, padding, next_layer_padding, layer_num;
};

int64_t pad_to(int64_t x, int64_t m) { return ((x + m - 1) / m) * m; }

}  // namespace

// Dorado SUP encoder on koi, variable chunk sizes (Dorado's default "VCS" path on sm80+):
// ConvStackImpl::run_koi_vcs_tx + TxEncoderStackImpl::forward(aux) + TxEncoderImpl::koi_forward,
// with the AuxiliaryData / CudaModelRunner::accept_chunk bookkeeping.
class KoiVcsEncoder {
public:
    // convs: per layer [weight (C_out, C_in, W), bias]; conv_params: per layer [winlen, stride].
    // tx_layers: per layer [Wqkv, out_proj.weight, out_proj.bias, norm1.weight, norm2.weight,
    //                       fc1.weight, fc2.weight]. All fp16 CUDA tensors.
    KoiVcsEncoder(const std::vector<std::vector<at::Tensor>> &convs,
                  const std::vector<std::vector<int64_t>> &conv_params,
                  const std::vector<std::vector<at::Tensor>> &tx_layers,
                  const std::vector<double> &alphas,
                  int64_t nhead,
                  int64_t dim_feedforward,
                  double theta,
                  int64_t max_seq_len,
                  int64_t max_chunk_size,
                  bool use_i8,
                  int64_t remove_bits)
            : H(nhead), max_chunk_size(max_chunk_size), use_i8(use_i8) {
        if (koi_tc_is_available(KOI_F16) != KOI_SUCCESS) {
            throw std::runtime_error("koi tiled path not available on this device");
        }
        if (koi_tc_is_available(KOI_E4M3) == KOI_SUCCESS) {
            throw std::runtime_error("fp8-capable device: Dorado would use the fp8 path, not ported");
        }
        // ConvStack: weights tiled as in ConvLayer::run_koi_vcs_tx
        stride_in = 1;
        for (size_t i = 0; i < convs.size(); ++i) {
            Conv c;
            const auto &w = convs[i].at(0);
            c.C_out = w.size(0);
            c.C_in = w.size(1);
            c.winlen = conv_params[i].at(0);
            c.stride = conv_params[i].at(1);
            c.padding = c.winlen / 2;
            c.layer_num = static_cast<int>(i);
            c.next_layer_padding = (i + 1 < convs.size()) ? conv_params[i + 1].at(0) / 2 : 0;
            auto wd = w.transpose(1, 2).contiguous().to(torch::kF16);
            if (i == 0) {
                wd = wd.view({c.C_out / 2, 2, c.winlen * c.C_in}).transpose(1, 2).contiguous();
            } else {
                const int SMEM_N = (i == 4) ? 128 : c.C_out;
                wd = wd.view({c.C_out / SMEM_N, SMEM_N / 16, 2, 8, (c.winlen * c.C_in) / 16, 2, 8})
                             .permute({0, 4, 1, 2, 5, 3, 6})
                             .contiguous();
            }
            c.w_device = wd;
            c.b_device = convs[i].at(1).to(torch::kF16);
            max_conv_padding = std::max<int64_t>(max_conv_padding, c.padding);
            stride_in *= c.stride;
            conv_vec.push_back(std::move(c));
        }
        granularity = stride_in * 64;  // BasecallModelConfig::chunk_size_granularity()

        const auto &w0 = tx_layers.at(0).at(0);
        C = w0.size(1);
        D = C / H;
        E = dim_feedforward * 2;
        auto f16_opts = w0.options().dtype(torch::kF16);

        // RotaryEmbeddingImpl: buffers computed in the model's options (fp16 on CUDA).
        std::vector<double> vec;
        for (float i = 0; i < D; i += 2) {
            vec.push_back(std::pow(theta, static_cast<double>(i / (float)D)));
        }
        at::Tensor inv_freq = torch::from_blob(vec.data(), {(int64_t)vec.size()},
                                               torch::TensorOptions().dtype(torch::kDouble))
                                      .to(f16_opts)
                                      .reciprocal();
        at::Tensor freqs =
                torch::arange(max_seq_len, f16_opts).reshape({max_seq_len, 1, 1, 1}) * inv_freq;
        at::Tensor cos_freqs = torch::cos(freqs).to(f16_opts);
        at::Tensor sin_freqs = torch::sin(freqs).to(f16_opts);
        const int64_t max_T = 16 * (max_seq_len / 16);
        if (max_T < max_chunk_size / stride_in) {
            throw std::runtime_error("RoPE table shorter than the maximum chunk");
        }
        sincos_bfr = torch::empty({max_T, D / 2, 2}, f16_opts);
        sincos_bfr.select(2, 0) = sin_freqs.slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr.select(2, 1) = cos_freqs.slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr = sincos_bfr.view({max_T / 16, 16, D / 8, 8}).transpose(1, 2).contiguous();

        for (size_t l = 0; l < tx_layers.size(); ++l) {
            const auto &w = tx_layers[l];
            Layer L;
            L.alpha = static_cast<float>(alphas.at(l));
            const auto &wqkv = w.at(0);
            L.wqkv_f16.t = torch::empty_like(wqkv);
            auto wqk = wqkv.view({3, H, 2, D / 2, C}).slice(0, 0, 2);
            L.wqkv_f16.t.view({3, H, D / 2, 2, C}).slice(0, 0, 2) = wqk.transpose(2, 3);
            L.wqkv_f16.t = L.wqkv_f16.t.view({3, H, D, C});
            L.wqkv_f16.t[2] = wqkv.view({3, H, D, C})[2];

            L.wqkv_i8 = quantize_tensor(L.wqkv_f16.t, -1);
            L.wqkv_i8.scale.reciprocal_();
            L.wqkv_i8.t =
                    L.wqkv_i8.t.view({3, H, D / 16, 16, C / 16, 16}).transpose(-3, -2).contiguous();
            L.wqkv_f16.t =
                    L.wqkv_f16.t.view({3, H, D / 16, 16, C / 8, 8}).transpose(-3, -2).contiguous();

            L.proj_weight = w.at(1).view({C / 16, 16, C / 8, 8}).transpose(1, 2).contiguous();
            L.proj_bias = w.at(2).view({C}).clone();
            L.res_weights = w.at(3).view({C}).clone();
            L.res2_weights = w.at(4).view({C}).clone();

            auto fc1_weight_interleaved =
                    w.at(5).unflatten(0, {2, -1, 16}).transpose(0, 1).contiguous();
            L.fc1_i8 = quantize_tensor(fc1_weight_interleaved, -1);
            L.fc1_i8.scale.reciprocal_();
            L.fc1_i8.t = L.fc1_i8.t.view({E / 16, 16, C / 16, 16}).transpose(1, 2).contiguous();
            L.fc2_wts = w.at(6).view({C / 16, 16, E / 16, 8}).transpose(1, 2).contiguous();
            L.fc1_f16.t =
                    fc1_weight_interleaved.view({E / 16, 16, C / 8, 8}).transpose(1, 2).contiguous();

            // TxEncoderImpl::remove_bits (default 4 bits for every tensor)
            const int rb = static_cast<int>(remove_bits);
            if (rb > 0) {
                apply_rounding(L.wqkv_f16.t, rb);
                apply_rounding(L.proj_weight, rb);
                apply_rounding(L.res_weights, rb);
                apply_rounding(L.fc2_wts, rb);
                apply_rounding(L.fc1_f16.t, rb);
                apply_rounding(L.res2_weights, rb);
            }
            layer_vec.push_back(std::move(L));
        }
    }

    int64_t granularity_samples() const { return granularity; }
    int64_t stride() const { return stride_in; }

    // signal: 1D fp16 CUDA, the chunks concatenated; lens: samples per chunk, each a multiple of
    // the granularity (768) and <= max_chunk_size. Returns frames [sum(lens) / stride, C] fp16,
    // the chunks' frames concatenated in order.
    at::Tensor forward(const at::Tensor &signal, const std::vector<int64_t> &lens) {
        const int n = static_cast<int>(lens.size());
        auto opts = signal.options();
        auto i32 = opts.dtype(torch::kInt32);
        auto f16_opts = opts.dtype(torch::kF16);
        auto stream = c10::cuda::getCurrentCUDAStream().stream();

        // AuxiliaryData: chunk table in granularity blocks, uploaded scaled to samples.
        std::vector<int32_t> table(2 * n);
        int64_t total_gran = 0;
        for (int i = 0; i < n; ++i) {
            if (lens[i] <= 0 || lens[i] % granularity != 0 || lens[i] > max_chunk_size) {
                throw std::runtime_error("KoiVcsEncoder: chunk length must be a positive multiple "
                                         "of the granularity and <= chunk size");
            }
            table[2 * i] = static_cast<int32_t>(total_gran);
            table[2 * i + 1] = static_cast<int32_t>(lens[i] / granularity);
            total_gran += lens[i] / granularity;
        }
        const int64_t T_enc = granularity / stride_in;  // 64 frames per block
        const int64_t p = 256 / T_enc;                  // koi MatMulOp: M multiple of 256
        const int64_t n_blocks = pad_to(total_gran, p);
        const int64_t max_num_granularity = n_blocks;
        auto chunk_table = torch::from_blob(table.data(), {n, 2}, torch::TensorOptions().dtype(torch::kInt32))
                                   .mul(granularity)
                                   .to(i32);
        auto conv_load_lut = torch::empty({total_gran}, i32);
        auto conv_store_lut = torch::empty({total_gran}, i32);
        auto qkv_rope_lut = torch::zeros({n_blocks}, i32);

        // CudaModelRunner::accept_chunk: [pad][chunk][pad][chunk]...[pad], pad = first conv padding
        const int64_t first_pad = conv_vec.front().padding;
        auto in = torch::zeros({n_blocks * granularity + (n_blocks + 1) * max_conv_padding}, f16_opts);
        int64_t off = first_pad, src = 0;
        for (int i = 0; i < n; ++i) {
            in.narrow(0, off, lens[i]).copy_(signal.narrow(0, src, lens[i]));
            off += lens[i] + first_pad;
            src += lens[i];
        }
        at::Tensor x = in.view({1, -1});

        int64_t gran = granularity;
        for (auto &c : conv_vec) {
            const int64_t M_input = x.size(1);
            int64_t M_out = M_input;
            if (c.stride > 1) {
                M_out = (c.layer_num != 4)
                                ? (M_input / c.stride) + (max_num_granularity * c.next_layer_padding)
                                : qkv_rope_lut.size(0) * T_enc;
            }
            auto out = torch::empty({M_out * c.C_out}, f16_opts);
            check(koi_vcs_tx_fill_conv_load_store_lut(
                          stream, n, chunk_table.data_ptr<int>(), conv_load_lut.data_ptr<int>(),
                          conv_store_lut.data_ptr<int>(), c.layer_num == 0 ? nullptr : x.data_ptr(),
                          (int)M_input, c.C_in, (int)gran, c.stride, c.padding, c.next_layer_padding),
                  "vcs_fill_conv_lut");
            check(koi_vcs_tx_cnn(stream, c.layer_num, x.data_ptr(), c.w_device.data_ptr(),
                                 out.data_ptr(), c.b_device.data_ptr(), conv_load_lut.data_ptr<int>(),
                                 conv_store_lut.data_ptr<int>(), (int)total_gran, (int)M_input,
                                 (int)M_out, c.winlen, c.C_in, c.C_out, c.padding, c.stride, 0),
                  "vcs_tx_cnn");
            gran /= c.stride;
            x = out.view({c.C_out / 8, M_out, 8});
        }

        // TxEncoderStackImpl::forward with aux
        const int64_t N = n_blocks, T = T_enc;
        check(koi_vcs_tx_fill_qkv_rope_lut(stream, n, (int)T, chunk_table.data_ptr<int>(),
                                           qkv_rope_lut.data_ptr<int>()),
              "vcs_fill_qkv_rope_lut");
        at::Tensor tiled_f16 = x.narrow(1, 0, N * T)
                                       .view({C / 8, N, T / 16, 16, 8})
                                       .permute({1, 2, 0, 3, 4})
                                       .contiguous();
        ScaledTensor scaled;
        if (use_i8) {
            scaled = quantize_tensor(x.narrow(1, 0, N * T)
                                             .view({C / 8, N * T, 8})
                                             .transpose(0, 1)
                                             .contiguous()
                                             .view({N * T, C}),
                                     1);
            scaled.scale.reciprocal_();
            scaled.t = scaled.t.view({N, T / 16, 16, C / 16, 16}).transpose(2, 3).contiguous();
        } else {
            scaled.t = tiled_f16;
        }
        for (auto &L : layer_vec) {
            layer_forward(L, scaled, tiled_f16, qkv_rope_lut, chunk_table, n);
        }
        return tiled_f16.transpose(2, 3).contiguous().view({N * T, C}).narrow(0, 0, total_gran * T);
    }

private:
    void layer_forward(Layer &L,
                       ScaledTensor &scaled_tensor,
                       at::Tensor &x_f16,
                       const at::Tensor &qkv_rope_lut,
                       const at::Tensor &chunk_table,
                       int n_chunks) {
        const int64_t N = x_f16.size(0);
        const int64_t T = x_f16.size(1) * 16;
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        auto ctr = torch::zeros({4}, x_f16.options().dtype(torch::kI32));
        auto f16_opts = x_f16.options().dtype(torch::kF16);
        auto f8_opts = x_f16.options().dtype(torch::kFloat8_e4m3fn);

        auto qkv = torch::empty({N, T / 16, 3, H, D / 8, 16, 8}, f16_opts);
        auto t_out_attn = torch::empty({N, T / 16, H, D / 8, 16, 8}, f16_opts);
        auto t_out_proj = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);
        auto t_rms_out = torch::empty({N, T / 16, C / 16, 16, 16}, f8_opts);
        auto t_fc1_out = torch::empty({N * T / 16, E / 16, 16, 8}, f16_opts);
        auto t_fc2_out = torch::empty({N, T / 16, C / 8, 16, 8}, f16_opts);

        auto &x = scaled_tensor.t;
        const bool i8 = (x.dtype() == torch::kI8);
        auto &wqkv_weights = i8 ? L.wqkv_i8 : L.wqkv_f16;
        auto &t_fc1_wts = i8 ? L.fc1_i8 : L.fc1_f16;
        KoiTensorExt in_f16(x_f16, {'N', 'T', 'C', 't', 'c'});
        KoiTensorExt in(x, {'N', 'T', 'C', 't', 'c'}, scaled_tensor.scale, 'C');
        KoiTensorExt *in_i8_ptr = i8 ? &in : nullptr;
        KoiTensorExt in_mk{x.flatten(0, 1), {'M', 'K', 'm', 'k'}, scaled_tensor.scale, 'K'};
        KoiTensorExt weights_qkv(wqkv_weights.t, {KOI_DIM_MAT_Q_K_V, 'H', 'D', 'C', 'd', 'c'},
                                 wqkv_weights.scale, 'C');
        KoiTensorExt out_qkv(qkv, {'N', 'T', KOI_DIM_MAT_Q_K_V, 'H', 'D', 't', 'd'});
        KoiTensorExt out_attn_mk(t_out_attn.flatten(2, 3).flatten(0, 1), {'M', 'K', 'm', 'k'});
        KoiTensorExt out_proj_mn(t_out_proj.flatten(0, 1), {'M', 'N', 'm', 'n'});
        KoiTensorExt out_proj_ntc(t_out_proj, {'N', 'T', 'C', 't', 'c'});
        KoiTensorExt fc1_out(t_fc1_out, {'M', 'N', 'm', 'n'});
        KoiTensorExt fc1_out_mk(t_fc1_out, {'M', 'K', 'm', 'k'});
        KoiTensorExt fc2_out_mn(t_fc2_out.flatten(0, 1), {'M', 'N', 'm', 'n'});
        KoiTensorExt fc2_out_ntc(t_fc2_out, {'N', 'T', 'C', 't', 'c'});
        KoiTensorExt sincos(sincos_bfr.slice(0, 0, (max_chunk_size / stride_in) / 16),
                            {'T', 'D', 't', 'd'});
        KoiTensorExt proj_w(L.proj_weight, {'N', 'K', 'n', 'k'});
        KoiTensorExt proj_b(L.proj_bias, {'N'});
        KoiTensorExt res_weights(L.res_weights, {'C'});
        KoiTensorExt res2_weights(L.res2_weights, {'C'});
        KoiTensorExt fc1_wts(t_fc1_wts.t, {'N', 'K', 'n', 'k'}, t_fc1_wts.scale, 'K');
        KoiTensorExt fc2_wts(L.fc2_wts, {'N', 'K', 'n', 'k'});

        check(koi_qkv_rotary(stream, &in, &weights_qkv, &sincos, &out_qkv,
                             qkv_rope_lut.data_ptr<int>(), ctr[0].data_ptr<int>()),
              "qkv_rotary");
        check(koi_vcs_attn(stream, qkv.data_ptr(), chunk_table.data_ptr<int>(), n_chunks,
                           t_out_attn.data_ptr()),
              "vcs_attn");
        check(koi_linear(stream, &out_attn_mk, &proj_w, &proj_b, &out_proj_mn,
                         ctr[1].data_ptr<int>()),
              "out_proj");
        check(koi_rmsnorm_residual(stream, &out_proj_ntc, &in_f16, L.alpha, &res_weights, &in_f16,
                                   nullptr, in_i8_ptr),
              "rmsnorm1");
        check(koi_mm_swiglu(stream, &in_mk, &fc1_wts, &fc1_out, ctr[2].data_ptr<int>(), 0, 0),
              "fc1_swiglu");
        check(koi_linear(stream, &fc1_out_mk, &fc2_wts, nullptr, &fc2_out_mn,
                         ctr[3].data_ptr<int>()),
              "fc2");
        check(koi_rmsnorm_residual(stream, &fc2_out_ntc, &in_f16, L.alpha, &res2_weights, &in_f16,
                                   nullptr, in_i8_ptr),
              "rmsnorm2");
    }

    int64_t H, C = 0, D = 0, E = 0, max_chunk_size, stride_in = 1, granularity = 0,
               max_conv_padding = 0;
    bool use_i8;
    at::Tensor sincos_bfr;
    std::vector<Conv> conv_vec;
    std::vector<Layer> layer_vec;
};

// dorado/basecall/decode/CUDADecoder.cpp: beam_search_part_1, fixed-size chunks (no aux).
// scores: [N, T, C] fp16 CUDA CRF scores (C = 4096 for SUP, blank score fixed).
// Returns [3, N, T] int8: moves, sequence (ASCII), qstring (ASCII).
at::Tensor beam_search(const at::Tensor &scores_in,
                       double post_clamp,
                       int64_t beam_width,
                       double beam_cut,
                       double blank_score,
                       double q_scale,
                       double q_shift,
                       bool move_pad) {
    auto scores = scores_in.contiguous();
    const long N = scores.size(0), T = scores.size(1), C = scores.size(2);
    auto i32 = scores.options().dtype(torch::kInt32);
    auto i8 = scores.options().dtype(torch::kChar);
    auto chunks = torch::empty({N, 4}, i32);
    chunks.select(1, 0) = torch::arange(0, int(T * N), int(T), i32);
    chunks.select(1, 2) = torch::arange(0, int(T * N), int(T), i32);
    chunks.select(1, 1) = int(T);
    chunks.select(1, 3) = 0;
    auto chunk_results = torch::empty({N, 8}, i32);
    auto aux = torch::empty({N * (T + 1) * (C + 4 * beam_width)}, i8);
    auto path = torch::zeros({N * (T + 1)}, i32);
    auto msq = torch::zeros({3, N * T}, i8);
    auto moves = msq[0], sequence = msq[1], qstring = msq[2];
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
#define KOI_DECODE_ARGS                                                                         \
    stream, chunks.data_ptr(), chunk_results.data_ptr(), int(N), scores.data_ptr(),            \
            float(post_clamp), int(C), aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL, \
            sequence.data_ptr(), qstring.data_ptr(), float(q_scale), float(q_shift),             \
            int(beam_width), float(beam_cut), float(blank_score)
    check(host_back_guide_step(KOI_DECODE_ARGS), "back_guide_step");
    check(host_beam_search_step(KOI_DECODE_ARGS), "beam_search_step");
    check(host_compute_posts_step(KOI_DECODE_ARGS), "compute_posts_step");
    check(host_run_decode(KOI_DECODE_ARGS, int(move_pad)), "run_decode");
#undef KOI_DECODE_ARGS
    return msq.reshape({3, N, -1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beam_search", &beam_search);
    py::class_<KoiVcsEncoder>(m, "KoiVcsEncoder")
            .def(py::init<const std::vector<std::vector<at::Tensor>> &,
                          const std::vector<std::vector<int64_t>> &,
                          const std::vector<std::vector<at::Tensor>> &, const std::vector<double> &,
                          int64_t, int64_t, double, int64_t, int64_t, bool, int64_t>())
            .def("forward", &KoiVcsEncoder::forward)
            .def("granularity", &KoiVcsEncoder::granularity_samples)
            .def("stride", &KoiVcsEncoder::stride);
}
