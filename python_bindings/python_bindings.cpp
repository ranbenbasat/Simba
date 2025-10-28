#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include "../ExactQUIVER.h"
#include "../ApproxQUIVER.h"
#include "../Simba.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_exact_accelerated(torch::Tensor svec, int s) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ExactQUIVER<false, true> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr);
    auto Q = A.calcQuantizationValues(s);
    auto tQ = torch::zeros(Q.size(), torch::kFloat64);
    memcpy(tQ.data_ptr(), Q.data(), sizeof(double) * Q.size());

    return tQ;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_approx(torch::Tensor svec, int s, int m) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ApproxQUIVER<false> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr, m);
    auto Q = A.calcQuantizationValues(s);
    auto tQ = torch::zeros(Q.size(), torch::kFloat64);
    memcpy(tQ.data_ptr(), Q.data(), sizeof(double) * Q.size());

    return tQ;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor simba(torch::Tensor svec, int s, int l, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, int m_quiver, bool debug, string log_cost_fn = "") {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    Simba S;
    double* initial_levels = new double[s];
    if (m_quiver == -1) { // Using (exact) QUIVER
        auto initial_levels = quiver_exact_accelerated(svec, s);
        if (iters == 0) {
            return torch::zeros(s, torch::kFloat64);
        }
        auto Q = S.calcQuantizationValues(svec.data_ptr<double>(), svec.numel(), s, l, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, debug, initial_levels.data_ptr<double>(), log_cost_fn);
        if (l > 1) {
            auto tQ = torch::zeros({l, s}, torch::kFloat64);
            for (int i = 0; i < l; ++i) {
                memcpy(tQ[i].data_ptr(), Q[i].data(), sizeof(double) * s);
            }
            return tQ;
        }
        auto tQ = torch::zeros(s, torch::kFloat64);
        memcpy(tQ.data_ptr(), Q[0].data(), sizeof(double) * s);
        return tQ;
    } else { // Using Apx. QUIVER
        auto initial_levels = quiver_approx(svec, s, m_quiver);
        if (iters == 0) {
            return torch::zeros(s, torch::kFloat64);
        }
        auto Q = S.calcQuantizationValues(svec.data_ptr<double>(), svec.numel(), s, l, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, debug, initial_levels.data_ptr<double>(), log_cost_fn);
        if (l > 1) {
            auto tQ = torch::zeros({l, s}, torch::kFloat64);
            for (int i = 0; i < l; ++i) {
                memcpy(tQ[i].data_ptr(), Q[i].data(), sizeof(double) * s);
            }
            return tQ;
        }
        auto tQ = torch::zeros(s, torch::kFloat64);
        memcpy(tQ.data_ptr(), Q[0].data(), sizeof(double) * s);
        return tQ;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor asq_vnmse(torch::Tensor X, torch::Tensor Q, py::object W = py::none()) {
    TORCH_CHECK(X.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(X.size(-1) == X.numel(), "the input vector must be 1D");
    TORCH_CHECK(X.is_contiguous(), "the input vector must be contiguous in memory");
    
    TORCH_CHECK(Q.device().type() == torch::kCPU, "the quantization vector must be a kCPU tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat64, "the quantization vector must be a double (kFloat64)");
    TORCH_CHECK(Q.size(-1) == Q.numel(), "the quantization vector must be 1D");
    TORCH_CHECK(Q.is_contiguous(), "the quantization vector must be contiguous in memory");
    
    std::vector<double> X_vec(X.data_ptr<double>(), X.data_ptr<double>() + X.numel());
    std::vector<double> Q_vec(Q.data_ptr<double>(), Q.data_ptr<double>() + Q.numel());
    
    double result;
    if (!W.is_none()) {
        torch::Tensor W_tensor = W.cast<torch::Tensor>();
        TORCH_CHECK(W_tensor.device().type() == torch::kCPU, "the weight vector must be a kCPU tensor");
        TORCH_CHECK(W_tensor.dtype() == torch::kFloat64, "the weight vector must be a double (kFloat64)");
        TORCH_CHECK(W_tensor.size(-1) == W_tensor.numel(), "the weight vector must be 1D");
        TORCH_CHECK(W_tensor.is_contiguous(), "the weight vector must be contiguous in memory");
        std::vector<double> W_vec(W_tensor.data_ptr<double>(), W_tensor.data_ptr<double>() + W_tensor.numel());
        result = sq_vnmse(X_vec, Q_vec, &W_vec);
    } else {
        result = sq_vnmse(X_vec, Q_vec, nullptr);
    }
    
    return torch::tensor(result, torch::kFloat64);
}

torch::Tensor auq_vnmse(torch::Tensor X, torch::Tensor Q, double snorm = -1.0) {
    TORCH_CHECK(X.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(X.size(-1) == X.numel(), "the input vector must be 1D");
    TORCH_CHECK(X.is_contiguous(), "the input vector must be contiguous in memory");
    
    TORCH_CHECK(Q.device().type() == torch::kCPU, "the quantization matrix must be a kCPU tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat64, "the quantization matrix must be a double (kFloat64)");
    TORCH_CHECK(Q.is_contiguous(), "the quantization matrix must be contiguous in memory");
    
    std::vector<double> X_vec(X.data_ptr<double>(), X.data_ptr<double>() + X.numel());
    
    // Convert Q tensor to vector<vector<double>>
    std::vector<std::vector<double>> Q_vec;
    if (Q.dim() == 1) {
        // Single quantization vector
        Q_vec.resize(1);
        Q_vec[0] = std::vector<double>(Q.data_ptr<double>(), Q.data_ptr<double>() + Q.numel());
    } else if (Q.dim() == 2) {
        // Multiple quantization vectors
        int rows = Q.size(0);
        int cols = Q.size(1);
        Q_vec.resize(rows);
        for (int i = 0; i < rows; ++i) {
            Q_vec[i] = std::vector<double>(Q[i].data_ptr<double>(), Q[i].data_ptr<double>() + cols);
        }
    } else {
        TORCH_CHECK(false, "Q must be 1D or 2D tensor");
    }
    
    double result = calc_SR_vNMSE(X_vec, Q_vec, snorm);
    return torch::tensor(result, torch::kFloat64);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quiver_exact_accelerated", &quiver_exact_accelerated, 
          py::arg("svec"), py::arg("s"));
    
    m.def("quiver_approx", &quiver_approx, 
          py::arg("svec"), py::arg("s"), py::arg("m"));
    
    m.def("simba", &simba, 
          py::arg("svec"), py::arg("s"), py::arg("l"), py::arg("iters"), 
          py::arg("bin_iters"), py::arg("bin_iters_increase_threshold"), 
          py::arg("stopping_threshold"), py::arg("m_quiver"), 
          py::arg("debug") = false, py::arg("log_cost_fn") = "");
    
    m.def("asq_vnmse", &asq_vnmse, 
          py::arg("X"), py::arg("Q"), py::arg("W") = py::none());
    
    m.def("auq_vnmse", &auq_vnmse, 
          py::arg("X"), py::arg("Q"), py::arg("snorm") = -1.0);
}
