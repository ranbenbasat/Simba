#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include "../../ExactQUIVER.h"
#include "../../ApproxQUIVER.h"
#include "../../Simba.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_exact_accelerated(torch::Tensor svec, int s) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ExactQUIVER<false, true> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_approx(torch::Tensor svec, int s, int m) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ApproxQUIVER<false> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr, m);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor simba(torch::Tensor svec, int s, int q, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, int m_quiver, int m_simba, bool debug, string quantype, string log_cost_fn = "") {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

	Simba A;
	double* initial_levels = new double[s];
	if (m_quiver == -1) { // Using (exact) QUIVER
		
		auto initial_levels = quiver_exact_accelerated(svec, s);
		if (iters == 0){
			return torch::zeros(s, torch::kFloat64);
		}
		auto sqv = A.calcQuantizationValues(svec.data_ptr<double>(), svec.numel(), s, q, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, m_simba, debug, quantype, initial_levels.data_ptr<double>(), log_cost_fn);
		if (q > 1){
			auto tsqv = torch::zeros({q, s}, torch::kFloat64);
			for (int i = 0; i < q; ++i)
			{
				memcpy(tsqv[i].data_ptr(), sqv[i].data(), sizeof(double) * s);
			}
			return tsqv;
		}
		auto tsqv = torch::zeros(s, torch::kFloat64);
		memcpy(tsqv.data_ptr(), sqv[0].data(), sizeof(double) * s);		
		return tsqv;
	}
	else {// Using Apx. QUIVER		
		auto initial_levels = quiver_approx(svec, s, m_quiver);
		if (iters == 0){
			return torch::zeros(s, torch::kFloat64);
		}
		auto sqv = A.calcQuantizationValues(svec.data_ptr<double>(), svec.numel(), s, q, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, m_simba, debug, quantype, initial_levels.data_ptr<double>(), log_cost_fn);
		if (q > 1){
			auto tsqv = torch::zeros({q, s}, torch::kFloat64);
			for (int i = 0; i < q; ++i)
			{
				memcpy(tsqv[i].data_ptr(), sqv[i].data(), sizeof(double) * s);
			}
			return tsqv;
		}
		auto tsqv = torch::zeros(s, torch::kFloat64);
		memcpy(tsqv.data_ptr(), sqv[0].data(), sizeof(double) * s);		
		return tsqv;
	}
}




//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quiver_exact_accelerated", &quiver_exact_accelerated, "quiver_exact_accelerated");
    m.def("quiver_approx", &quiver_approx, "quiver_approx");
    m.def("simba", &simba, "simba");
}

