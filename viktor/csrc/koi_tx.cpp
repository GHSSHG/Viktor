// Dorado SUP transformer stack on Dorado's koi kernels (fixed-size chunks, no VCS).
//
// Port of dorado/nn/TxModules.cpp (TxEncoderStackImpl::forward with aux == nullptr and
// TxEncoderImpl::koi_forward), dorado commit 8b8fc5d: same weight tiling, int8 quantisation,
// remove_bits rounding, fp16 RoPE tables and koi call sequence. The fp8 path (Hopper only) is
// not ported; it is rejected at construction.

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

}  // namespace

class KoiTxStack {
public:
    // layers: per layer [Wqkv, out_proj.weight, out_proj.bias, norm1.weight, norm2.weight,
    //                    fc1.weight, fc2.weight], fp16 CUDA tensors.
    KoiTxStack(const std::vector<std::vector<at::Tensor>> &layers,
               const std::vector<double> &alphas,
               int64_t nhead,
               int64_t dim_feedforward,
               int64_t win_upper,
               int64_t win_lower,
               double theta,
               int64_t max_seq_len,
               bool use_i8,
               int64_t remove_bits,
               bool swiglu_f32_accum)
            : H(nhead),
              win_upper(win_upper),
              win_lower(win_lower),
              use_i8(use_i8),
              swiglu_f32_accum(swiglu_f32_accum) {
        if (koi_tc_is_available(KOI_F16) != KOI_SUCCESS) {
            throw std::runtime_error("koi tiled path not available on this device");
        }
        if (koi_tc_is_available(KOI_E4M3) == KOI_SUCCESS) {
            throw std::runtime_error("fp8-capable device: Dorado would use the fp8 path, not ported");
        }
        const auto &w0 = layers.at(0).at(0);
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
        max_T = 16 * (max_seq_len / 16);
        sincos_bfr = torch::empty({max_T, D / 2, 2}, f16_opts);
        sincos_bfr.select(2, 0) = sin_freqs.slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr.select(2, 1) = cos_freqs.slice(0, 0, max_T).view({max_T, D / 2});
        sincos_bfr = sincos_bfr.view({max_T / 16, 16, D / 8, 8}).transpose(1, 2).contiguous();

        for (size_t l = 0; l < layers.size(); ++l) {
            const auto &w = layers[l];
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

    // x: [N, T, C] fp16 CUDA, contiguous. Returns [N, T, C] fp16.
    at::Tensor forward(const at::Tensor &x) {
        const int64_t N = x.size(0), T = x.size(1);
        if (x.size(2) != C || T % 16 != 0 || T > max_T) {
            throw std::runtime_error("KoiTxStack: unsupported input shape");
        }
        at::Tensor tiled_f16 = x.view({N, T / 16, 16, C / 8, 8}).transpose(2, 3).contiguous();
        ScaledTensor scaled;
        if (use_i8) {
            scaled = quantize_tensor(x, 2);
            scaled.scale.reciprocal_();
            scaled.t = scaled.t.view({N, T / 16, 16, C / 16, 16}).transpose(2, 3).contiguous();
        } else {
            scaled.t = tiled_f16;
        }
        for (auto &L : layer_vec) {
            layer_forward(L, scaled, tiled_f16);
        }
        return tiled_f16.transpose(2, 3).contiguous().view({N, T, C});
    }

private:
    void layer_forward(Layer &L, ScaledTensor &scaled_tensor, at::Tensor &x_f16) {
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
        KoiTensorExt out_attn(t_out_attn, {'N', 'T', 'H', 'D', 't', 'd'});
        KoiTensorExt out_attn_mk(t_out_attn.flatten(2, 3).flatten(0, 1), {'M', 'K', 'm', 'k'});
        KoiTensorExt out_proj_mn(t_out_proj.flatten(0, 1), {'M', 'N', 'm', 'n'});
        KoiTensorExt out_proj_ntc(t_out_proj, {'N', 'T', 'C', 't', 'c'});
        KoiTensorExt fc1_out(t_fc1_out, {'M', 'N', 'm', 'n'});
        KoiTensorExt fc1_out_mk(t_fc1_out, {'M', 'K', 'm', 'k'});
        KoiTensorExt fc2_out_mn(t_fc2_out.flatten(0, 1), {'M', 'N', 'm', 'n'});
        KoiTensorExt fc2_out_ntc(t_fc2_out, {'N', 'T', 'C', 't', 'c'});
        KoiTensorExt sincos(sincos_bfr.slice(0, 0, T / 16), {'T', 'D', 't', 'd'});
        KoiTensorExt proj_w(L.proj_weight, {'N', 'K', 'n', 'k'});
        KoiTensorExt proj_b(L.proj_bias, {'N'});
        KoiTensorExt res_weights(L.res_weights, {'C'});
        KoiTensorExt res2_weights(L.res2_weights, {'C'});
        KoiTensorExt fc1_wts(t_fc1_wts.t, {'N', 'K', 'n', 'k'}, t_fc1_wts.scale, 'K');
        KoiTensorExt fc2_wts(L.fc2_wts, {'N', 'K', 'n', 'k'});

        check(koi_qkv_rotary(stream, &in, &weights_qkv, &sincos, &out_qkv, nullptr,
                             ctr[0].data_ptr<int>()),
              "qkv_rotary");
        check(koi_masked_attention(stream, (int)win_upper, (int)win_lower, &out_qkv, &out_attn),
              "masked_attention");
        check(koi_linear(stream, &out_attn_mk, &proj_w, &proj_b, &out_proj_mn,
                         ctr[1].data_ptr<int>()),
              "out_proj");
        check(koi_rmsnorm_residual(stream, &out_proj_ntc, &in_f16, L.alpha, &res_weights, &in_f16,
                                   nullptr, in_i8_ptr),
              "rmsnorm1");
        check(koi_mm_swiglu(stream, &in_mk, &fc1_wts, &fc1_out, ctr[2].data_ptr<int>(),
                            int(swiglu_f32_accum), 0),
              "fc1_swiglu");
        check(koi_linear(stream, &fc1_out_mk, &fc2_wts, nullptr, &fc2_out_mn,
                         ctr[3].data_ptr<int>()),
              "fc2");
        check(koi_rmsnorm_residual(stream, &fc2_out_ntc, &in_f16, L.alpha, &res2_weights, &in_f16,
                                   nullptr, in_i8_ptr),
              "rmsnorm2");
    }

    int64_t H, C = 0, D = 0, E = 0, max_T = 0, win_upper, win_lower;
    bool use_i8, swiglu_f32_accum;
    at::Tensor sincos_bfr;
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
    py::class_<KoiTxStack>(m, "KoiTxStack")
            .def(py::init<const std::vector<std::vector<at::Tensor>> &, const std::vector<double> &,
                          int64_t, int64_t, int64_t, int64_t, double, int64_t, bool, int64_t, bool>())
            .def("forward", &KoiTxStack::forward);
}
