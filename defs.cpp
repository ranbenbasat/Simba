#include "defs.h"

#include <cstdint>
#include <iostream>
#include <iomanip>



double sq_mse(vector<double> svec, vector<double> sqv, vector<double>* W = nullptr)
{
	int curr_sqv_index = 0;
	double mse = 0;


	for (int i = 0; i < svec.size(); ++i)
	{
		while (svec[i] > sqv[curr_sqv_index + 1])
		{
			curr_sqv_index++;
		}
		double w = W ? (*W)[i] : 1;
		mse += (svec[i] - sqv[curr_sqv_index]) * (sqv[curr_sqv_index + 1] - svec[i]) * w;
	}

	return mse;
}




double sq_vnmse(vector<double>& svec, vector<double>& sqv, vector<double>* W = nullptr)
{
	double snorm = 0;

	if (W) {
		for (int i = 0; i < svec.size(); ++i)
		{
			snorm += svec[i] * svec[i] * (*W)[i];
		}
	}
	else {
		for (int i = 0; i < svec.size(); ++i)
		{
			snorm += svec[i] * svec[i];
		}
	}

	return sq_mse(svec, sqv, W) / snorm;
}





void newCalcGrid(double* svec, int d, int M, string type, double* grid)
{
	//vector<double> grid(M);

	if (type == "Quantiles") {
		for (uint64_t i = 0; i < M; ++i)
		{
			int idx = (i * (d - 1)) / (M - 1);
			grid[i] = svec[idx];
		}
	}
	else if (type == "Uniform") {
		int coord_idx = -1;
		double min = svec[0];
		double max = svec[d - 1];
		double step = (max - min) / (M - 1);
		for (uint64_t i = 0; i < M; ++i)
		{
			while (svec[++coord_idx] < min + step * i) {
				if (d - coord_idx <= M - i) { // Picking all the remaining points
					for (; i < M; ++i) {
						grid[i] = svec[coord_idx++];
					}
					return;
				}
			}
			grid[i] = svec[coord_idx];
		}
	}

	return;
}

// Main function: computes vMSE from vec and sqv.
// If snorm is negative, it is computed as the squared norm of vec.
double calc_SR_vNMSE(const std::vector<double>& vec,
	const std::vector<std::vector<double>>& sqv,
	double snorm) {
	std::vector<double> thresholds = calcThresholds(sqv);
	int q = sqv.size();
	int s = sqv[0].size();
	int d = vec.size();

	// Compute bucket indices for each vec element using lower_bound.
	std::vector<int> buckets(d, 0), H(d, 0), S(d, 0);
	for (int i = 0; i < d; ++i) {
		// Find the first threshold not less than vec[i].
		auto it = std::lower_bound(thresholds.begin(), thresholds.end(), vec[i] - 1e-8);// this is a numeric correction that sometimes happens for the largest entry. it doesn't affect the calculation since it always rounds deterministically (for all Hs).
		int bucket = it - thresholds.begin();
		buckets[i] = bucket;
		H[i] = bucket % q;         // Row index
		S[i] = bucket / q;         // Column index
	}

	std::vector<double> det_sum(d, 0.0), det_err_sum(d, 0.0);
	// Loop over each possible H_prime and update sums.
	for (int H_prime = 0; H_prime < q; ++H_prime) {
		for (int i = 0; i < d; ++i) {
			if (H[i] > H_prime) {
				int col = S[i] + 1; // use S[i]+1 for indices where H > H_prime
				det_sum[i] += sqv[H_prime][col];
				double diff = vec[i] - sqv[H_prime][col];
				det_err_sum[i] += diff * diff;
			}
			else if (H[i] < H_prime) {
				int col = S[i]; // use S[i] for indices where H < H_prime
				det_sum[i] += sqv[H_prime][col];
				double diff = vec[i] - sqv[H_prime][col];
				det_err_sum[i] += diff * diff;
			}
		}
	}

	double sum_MSE = 0.0;
	// For each element compute x_p, p_up, sq_err and MSE.
	for (int i = 0; i < d; ++i) {
		double x_p = q * vec[i] - det_sum[i];
		int h = H[i];
		int s_idx = S[i];
		double denom = sqv[h][s_idx + 1] - sqv[h][s_idx];
		double p_up = (x_p - sqv[h][s_idx]) / denom;
		double sq_err = p_up * std::pow(sqv[h][s_idx + 1] - vec[i], 2) +
			(1 - p_up) * std::pow(sqv[h][s_idx] - vec[i], 2);
		double MSE = (det_err_sum[i] + sq_err) / q;
		sum_MSE += MSE;
	}

	double norm_sq = (snorm < 0) ? calcNormSquared(vec) : snorm;
	return sum_MSE / norm_sq;
}


// Helper: compute squared norm of a vector.
double calcNormSquared(const std::vector<double>& vec) {
	double sum = 0.0;
	for (double x : vec)
		sum += x * x;
	return sum;
}

// Compute thresholds from sqv.
// Assumes sqv is a q×s matrix.
std::vector<double> calcThresholds(const std::vector<std::vector<double>>& sqv) {
	int q = sqv.size();
	int s = sqv[0].size();
	std::vector<double> thresholds(q * (s - 1), 0.0);
	double T = 0.0;
	// Sum over sqv[j][0] for j=1,...,q-1 and add sqv[0][1]
	for (int j = 1; j < q; ++j)
		T += sqv[j][0];
	T += sqv[0][1];
	thresholds[0] = T;
	int idx = 1;
	for (int H = 1; H < q; ++H) {
		T += sqv[H][1] - sqv[H][0];
		thresholds[idx++] = T;
	}
	for (int X = 1; X < s - 1; ++X) {
		for (int H = 0; H < q; ++H) {
			T += sqv[H][X + 1] - sqv[H][X];
			thresholds[idx++] = T;
		}
	}
	// Divide each threshold by q.
	for (double& t : thresholds)
		t /= q;
	return thresholds;
}