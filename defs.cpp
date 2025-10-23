#include "defs.h"

#include <cstdint>
#include <iostream>
#include <iomanip>


// Compute thresholds from Q.
// Assumes Q is an ell�s matrix.
std::vector<double> calcThresholds(const std::vector<std::vector<double>>& Q) {
	int ell = Q.size();
	int s = Q[0].size();
	std::vector<double> thresholds(ell * (s - 1), 0.0);
	double T = 0.0;
	for (int h = 1; h < ell; ++h)
		T += Q[h][0];
	T += Q[0][1];
	thresholds[0] = T;
	int idx = 1;
	for (int h = 1; h < ell; ++h) {
		T += Q[h][1] - Q[h][0];
		thresholds[idx++] = T;
	}
	for (int q = 1; q < s - 1; ++q) {
		for (int h = 0; h < ell; ++h) {
			T += Q[h][q + 1] - Q[h][q];
			thresholds[idx++] = T;
		}
	}
	// Divide each threshold by ell.
	for (double& t : thresholds)
		t /= ell;
	return thresholds;
}

// Helper: compute squared norm of a vector.
double calcNormSquared(const std::vector<double>& vec) {
	double sum = 0.0;
	for (double x : vec)
		sum += x * x;
	return sum;
}

double sq_mse(vector<double> X, vector<double> Q, vector<double>* W = nullptr)
{
    int curr_sqv_index = 0;
    double mse = 0;
    for (int i = 0; i < X.size(); ++i) {
        while (X[i] > Q[curr_sqv_index + 1]) {
            curr_sqv_index++;
        }
        double w = W ? (*W)[i] : 1;
        mse += (X[i] - Q[curr_sqv_index]) * (Q[curr_sqv_index + 1] - X[i]) * w;
    }

    return mse;
}

double sq_vnmse(vector<double>& X, vector<double>& Q, vector<double>* W = nullptr)
{
    double snorm = 0;
    if (W) {
        for (int i = 0; i < X.size(); ++i) {
            snorm += X[i] * X[i] * (*W)[i];
        }
    } else {
        for (int i = 0; i < X.size(); ++i) {
            snorm += X[i] * X[i];
        }
    }
    return sq_mse(X, Q, W) / snorm;
}

// Main function: computes vMSE from vec and sqv.
// If snorm is negative, it is computed as the squared norm of vec.
double calc_SR_vNMSE(const std::vector<double>& X,
	const std::vector<std::vector<double>>& Q,
	double snorm) {
	std::vector<double> thresholds = calcThresholds(Q);
	int q = Q.size();
	int s = Q[0].size();
	int d = X.size();

	// Compute bucket indices for each vec element using lower_bound.
	std::vector<int> buckets(d, 0), H(d, 0), S(d, 0);
	for (int i = 0; i < d; ++i) {
		// Find the first threshold not less than vec[i].
		auto it = std::lower_bound(thresholds.begin(), thresholds.end(), X[i] - 1e-8);// this is a numeric correction that sometimes happens for the largest entry. it doesn't affect the calculation since it always rounds deterministically (for all Hs).
		int bucket = it - thresholds.begin();
		buckets[i] = bucket;
		H[i] = bucket % q;         // Row index
		S[i] = bucket / q;         // Column index
	}

	std::vector<double> det_sum(d, 0.0), det_err_sum(d, 0.0);
	// Loop over each possible H_prime and update sums.
	for (int h_prime = 0; h_prime < q; ++h_prime) {
		for (int i = 0; i < d; ++i) {
			if (H[i] > h_prime) {
				int col = S[i] + 1; // use S[i]+1 for indices where H > H_prime
				det_sum[i] += Q[h_prime][col];
				double diff = X[i] - Q[h_prime][col];
				det_err_sum[i] += diff * diff;
			}
			else if (H[i] < h_prime) {
				int col = S[i]; // use S[i] for indices where H < H_prime
				det_sum[i] += Q[h_prime][col];
				double diff = X[i] - Q[h_prime][col];
				det_err_sum[i] += diff * diff;
			}
		}
	}

    double sum_MSE = 0.0;
    // For each element compute x_p, p_up, sq_err and MSE.
	for (int i = 0; i < d; ++i) {
		double x_p = q * X[i] - det_sum[i];
		int h = H[i];
		int s_idx = S[i];
		double denom = Q[h][s_idx + 1] - Q[h][s_idx];
		double p_up = (x_p - Q[h][s_idx]) / denom;
		double sq_err = p_up * std::pow(Q[h][s_idx + 1] - X[i], 2) +
			(1 - p_up) * std::pow(Q[h][s_idx] - X[i], 2);
		double MSE = (det_err_sum[i] + sq_err) / q;
		sum_MSE += MSE;
	}

    double norm_sq = (snorm < 0) ? calcNormSquared(X) : snorm;
    return sum_MSE / norm_sq;
}
