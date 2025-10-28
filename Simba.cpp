#include "Simba.h"

/*
* Calculate the thresholds based on the current levels. If changed_h and changed_q are -1, calculate all thresholds from scratch, otherwise only update the thresholds affected by the change in levels[changed_h][changed_q].
*/
void Simba::calcThresholds(int changed_h, int changed_q)
{
    if (changed_q == -1) {
        double T = 0;
        for (int j = 1; j < ell; ++j) {
            T += Q[j][0];
        }
        T += Q[0][1];
        //new_threshold_indices[0] = upper_bound(svec, svec + d, T / ell) - svec - 1;
        new_threshold_indices[0] = upper_bound(X, X + d, T * one_plus_eps / ell) - X - 1; // multiplying by (1+eps) to avoid numerical errors. Don't remove!
        threshold_values[0] = T / ell;

        int idx = 0;
        for (int h = 1; h < ell; ++h) {
            T += Q[h][1] - Q[h][0];
            new_threshold_indices[++idx] = (upper_bound(X, X + d, T * one_plus_eps / ell) - X) - 1;
            threshold_values[idx] = T / ell;
        }
        for (int q = 1; q < s - 1; ++q) {
            for (int h = 0; h < ell; ++h) {
                T += Q[h][q + 1] - Q[h][q];
                new_threshold_indices[++idx] = (upper_bound(X, X + d, T * one_plus_eps / ell) - X) - 1;
                threshold_values[idx] = T / ell;
            }
        }
        new_threshold_indices[ell * (s - 1) - 1] = d - 1; //The last threshold should include the end of the vector, this avoids numerical issues.
    } else {
        int changed_j = changed_h + changed_q * ell;
        int first_changed_i = changed_j > ell ? changed_j - (ell - 1) : 1;
        double T = 0;
        for (int j = first_changed_i; j < first_changed_i + ell; ++j) {
            T += Q[j % ell][j / ell];
        }
        new_threshold_indices[first_changed_i - 1] = (upper_bound(X, X + d, T * one_plus_eps / ell) - X) - 1;
        new_threshold_indices[first_changed_i - 1] = new_threshold_indices[first_changed_i - 1] < 0 ? 0 : new_threshold_indices[first_changed_i - 1];
        threshold_values[first_changed_i - 1] = T / ell;
        int last_i = changed_j < ell * (s - 1) - 1 ? changed_j : ell * (s - 1) - 1;
        for (int i = first_changed_i + 1; i <= last_i; ++i) {
            int h = (i - 1) % ell;
            int q = (i - 1) / ell;
            T += Q[h][q + 1] - Q[h][q];
            new_threshold_indices[i - 1] = (upper_bound(X, X + d, T * one_plus_eps / ell) - X) - 1;
            threshold_values[i - 1] = T / ell;
        }
        new_threshold_indices[ell * (s - 1) - 1] = d - 1;
        threshold_values[ell * (s - 1) - 1] = maxX;
    }
}

void Simba::calcBetas() {
    int N = new_threshold_indices.size();
    if (N == 0) return;
    // compute betai[0] from scratch
    double sum = 0;
    for (int h = 0; h < ell; ++h) {
        int idx = (ell - h) / ell;  // same as (0 + ell - j) / ell
        double v = Q[h][idx];
        sum += v * v;
    }
    betai[0] = sum;
    // roll-forward: only j0 = (i+1)%ell changes its index
    for (int i = 0; i + 1 < N; ++i) {
        int j0 = (i + 1) % ell;
        int old_idx = (i + ell - j0) / ell;
        int new_idx = (i + 1 + ell - j0) / ell;
        double old_v = Q[j0][old_idx];
        double new_v = Q[j0][new_idx];
        betai[i + 1] = betai[i] + new_v * new_v - old_v * old_v;
    }
}

/*
* Update betas after changing levels[j][q] from oldval to newval.
*/
void Simba::updateBetas(int h, int q, double oldval, double newval) {
    int N = new_threshold_indices.size();
    // change in the squared term
    double diff = newval * newval - oldval * oldval;

    // solve (i + ell - j) / ell == q  =>  (q-1)*ell + j <= i < q*ell + j
    int i_start = (q - 1) * ell + h;
    int i_end = q * ell + h;
    // clamp to valid range
    if (i_start < 0)    i_start = 0;
    if (i_end > N)    i_end = N;
    // update only those betai[i] that include levels[j][q]^2
    for (int i = i_start; i < i_end; ++i) {
        betai[i] += diff;
    }
}

/*
* Calculate MSEi for interval i.
*/
double Simba::calc_MSEi(int i) const {
    int h_star = i % ell;
    int R_idx = (i + ell - h_star) / ell;
    int L_idx = R_idx - 1;
    double Ri = Q[h_star][R_idx];
    double Li = Q[h_star][L_idx];
    double Ti = threshold_values[i];
    double Ni = (i == 0) ? new_threshold_indices[i] + 1 : new_threshold_indices[i] - new_threshold_indices[i - 1];
    double betaival = betai[i];
    double pcumsum = (i == 0) ? pts_cumsum[new_threshold_indices[i]] : pts_cumsum[new_threshold_indices[i]] - pts_cumsum[new_threshold_indices[i - 1]];
    return (Ri + Li) * pcumsum + (betaival / ell - Ti * (Ri + Li)) * Ni;
}

/*
* Calculate the MSE cost. If changed_h and changed_q are -1, calculate from scratch, otherwise only update the affected intervals.
*/
double Simba::MSE(int changed_h = -1, int changed_q = -1) {
    calcThresholds(changed_h, changed_q);
    if (changed_q == -1) {
        double total_cost = 0;
        calcBetas();
        for (int i = 0; i < new_threshold_indices.size(); ++i) {
            MSEi[i] = calc_MSEi(i);
            total_cost += MSEi[i];
        }
        lastMSE = total_cost - sos;
    }
    else {
        int level_idx = changed_h + changed_q * ell;
        int min_changed_treshold_idx = level_idx - ell - 2;
        int max_changed_treshold_idx = level_idx + 1;
        if (min_changed_treshold_idx < 0)
            min_changed_treshold_idx = 0;
        if (max_changed_treshold_idx > new_threshold_indices.size())
            max_changed_treshold_idx = new_threshold_indices.size();
        for (int i = min_changed_treshold_idx; i < max_changed_treshold_idx; ++i) {
            lastMSE -= MSEi[i];
            MSEi[i] = calc_MSEi(i);
            lastMSE += MSEi[i];
        }
    }
    return lastMSE;
}

void Simba::preprocess()
{
    minX = X[0];
    maxX = X[d - 1];
    pts_cumsum.resize(d);
    pts_cumsum[0] = minX;
    sos = minX * minX;
    for (int i = 1; i < d; ++i) {
        pts_cumsum[i] = pts_cumsum[i - 1] + X[i];
        sos += X[i] * X[i];
    }
}

vector<vector<double>> Simba::_calcQuantizationValues(double* initial_levels, string log_cost)
{
    if (log_cost != "") {
        cout << "WARNING: Calculating quantization values with log_cost enabled." << endl;
    }
    new_threshold_indices.resize(ell * (s - 1));
    threshold_values = new double[ell * (s - 1)] {};
    MSEi = new double[ell * (s - 1)] {};
    betai = new double[ell * (s - 1)] {};

    preprocess();

    Q.resize(ell);
    if (initial_levels) {
        for (int h = 0; h < ell; ++h) {
            Q[h].resize(s);
            for (int j = 0; j < s; ++j)
                Q[h][j] = initial_levels[j];
        }
    }
    else {
        throw runtime_error("Initial levels must be provided.");
    }

    vector<pair<int, int>> changed_levels(s * ell - 2);
    {
        int idx = -1;
        if (ell == 1) {
            for (int y = 1; y < s - 1; ++y) {
                changed_levels[++idx] = make_pair(0, y);
            }
        }
        else {
            for (int x = 0; x < ell; ++x) {
                for (int y = 0; y < s; ++y) {
                    if (((x == 0) && (y > 0)) || ((x > 0) && (x < ell - 1)) || ((x == ell - 1) && (y < s - 1))) {
                        changed_levels[++idx] = make_pair(x, y);
                    }
                }
            }
        }

    }

    double old_cost = numeric_limits<double>::max();
    int stuck = 0;
    int totally_stuck = 0;
    double _cost = 0;

    _cost = MSE();
    min_required_improvement = _cost * (1 - stopping_threshold);
    vector<pair<double, int>> cost_vec;

    double lo, hi, mid, mid_plus_eps, cost_mid, cost_mid_plus_eps;
    int iter_num = 0;
    int h = 1, q = 0;
    bool last_changed_was_hi;
    for (; iter_num < iters; ++iter_num) {
        int remaining_bin_iters = bin_iters;
        _cost = MSE(h, q);
        h = changed_levels[iter_num % (s * ell - 2)].first;
        q = changed_levels[iter_num % (s * ell - 2)].second;

        if ((log_cost != "") && (iter_num % 1 == 0)) {
            cost_vec.push_back(make_pair(_cost, totally_stuck));
        }
        double orig_val = Q[h][q];

        if (iter_num % (s * ell) == 0) {
            if (debug)
                cout << iter_num << " " << _cost << endl;
            if (_cost > old_cost * bin_iters_increase_threshold) {
                stuck += 1;
                if (stuck >= 5) {
                    if (bin_iters < 10) {
                        bin_iters += 1;
                    }
                    stuck = 0;
                }
            }
            if (_cost > old_cost - min_required_improvement) {
                totally_stuck += 1;
                if (totally_stuck > 3)
                    break;
            }
            else {
                totally_stuck = 0;
            }
            old_cost = min(_cost, old_cost);
        }

        if (h == 0) {
            lo = Q[ell - 1][q - 1];
        }
        else {
            lo = Q[h - 1][q];
            if (q == 0) {
                lo = max(lo, Q[h][q] - (Q[1][0] - Q[0][0]));
                if (h == 1) {
                    lo = max(lo, (Q[1][0] + Q[0][0]) / 2);
                }
            }
        }
        if (h == ell - 1) {
            hi = Q[0][q + 1];
        }
        else {
            hi = Q[h + 1][q];
            if (q == s - 1) {
                hi = min(hi, Q[h][q] + (Q[ell - 1][s - 1] - Q[ell - 2][s - 1]));
                if (h == ell - 2) {
                    hi = min(hi, (Q[ell - 1][s - 1] + Q[ell - 2][s - 1]) / 2);
                }
            }
        }
        if (lo >= hi - 1e-7)
            continue;

        mid = (hi + lo) / 2;
        mid_plus_eps = mid + (hi - mid) * 1e-5;
        while (--remaining_bin_iters >= 0) {
            updateBetas(h, q, Q[h][q], mid);
            Q[h][q] = mid;
            if (q == 0) {
                double res = ell * minX;
                for (int i = 1; i < ell; ++i) {
                    res -= Q[i][0];
                }
                updateBetas(0, 0, Q[0][0], res);
                Q[0][0] = res;
            }
            else if (q == s - 1) {
                double res = ell * maxX;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= Q[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, Q[ell - 1][s - 1], res);
                Q[ell - 1][s - 1] = res;
            }
            cost_mid = MSE(h, q);

            updateBetas(h, q, Q[h][q], mid_plus_eps);
            Q[h][q] = mid_plus_eps;
            if (q == 0) {
                double res = ell * minX;
                for (int i = 1; i < ell; ++i) {
                    res -= Q[i][0];
                }
                updateBetas(0, 0, Q[0][0], res);
                Q[0][0] = res;
            }
            else if (q == s - 1) {
                double res = ell * maxX;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= Q[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, Q[ell - 1][s - 1], res);
                Q[ell - 1][s - 1] = res;
            }
            cost_mid_plus_eps = MSE(h, q);
            if (cost_mid < cost_mid_plus_eps) {
                hi = mid;
                last_changed_was_hi = true;
            }
            else {
                lo = mid;
                last_changed_was_hi = false;
            }
            mid = (hi + lo) / 2;
            mid_plus_eps = mid + (hi - mid) * 1e-5;
        }
        double cost_lo, cost_hi;
        if (last_changed_was_hi)
        {
            updateBetas(h, q, Q[h][q], lo);
            Q[h][q] = lo;
            if (q == 0) {
                double res = ell * minX;
                for (int i = 1; i < ell; ++i) {
                    res -= Q[i][0];
                }
                updateBetas(0, 0, Q[0][0], res);
                Q[0][0] = res;
            }
            else if (q == s - 1) {
                double res = ell * maxX;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= Q[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, Q[ell - 1][s - 1], res);
                Q[ell - 1][s - 1] = res;
            }
            cost_lo = MSE(h, q);
            cost_hi = cost_mid;
        }
        else
        {
            updateBetas(h, q, Q[h][q], hi);
            Q[h][q] = hi;
            if (q == 0) {
                // this can be optimized!
                double res = ell * minX;
                for (int i = 1; i < ell; ++i) {
                    res -= Q[i][0];
                }
                updateBetas(0, 0, Q[0][0], res);
                Q[0][0] = res;
            }
            else if (q == s - 1) {
                double res = ell * maxX;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= Q[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, Q[ell - 1][s - 1], res);
                Q[ell - 1][s - 1] = res;
            }
            cost_hi = MSE(h, q);
            cost_lo = cost_mid;
        }

        if ((_cost > cost_lo) || (_cost > cost_hi)) {
            double newval = (cost_lo < cost_hi) ? lo : hi;
            updateBetas(h, q, Q[h][q], newval);
            Q[h][q] = newval;
        }
        else {
            updateBetas(h, q, Q[h][q], orig_val);
            Q[h][q] = orig_val;
        }
        if (q == 0) {
            double res = ell * minX;
            for (int i = 1; i < ell; ++i) {
                res -= Q[i][0];
            }
            updateBetas(0, 0, Q[0][0], res);
            Q[0][0] = res;
        }
        else if (q == s - 1) {
            double res = ell * maxX;
            for (int i = 0; i < ell - 1; ++i) {
                res -= Q[i][s - 1];
            }
            updateBetas(ell - 1, s - 1, Q[ell - 1][s - 1], res);
            Q[ell - 1][s - 1] = res;
        }
    }

    if (debug) {
        for (int h = 0; h < ell; ++h) {
            for (int q = 0; q < s - 1; ++q) {
                double res = 0;
                for (int h_prime = h + 1; h_prime < ell; ++h_prime) {
                    res -= Q[h_prime][q];
                }
                for (int h_prime = 0; h_prime < h; ++h_prime) {
                    res -= Q[h_prime][q + 1];
                }
            }
        }

        double norm2 = 0;
        for (int i = 0; i < d; ++i) {
            norm2 += X[i] * X[i];
        }

        double  __cost = MSE();
        cout << "Final cost = " << __cost << ", vNMSE = " << __cost / norm2 << endl;
    }

    delete[] threshold_values;
    delete[] MSEi;
    delete[] betai;

    if (log_cost != "") {
        std::ofstream ofs(log_cost);
        if (!ofs) throw std::runtime_error("Failed to open " + log_cost);
        for (const auto& x : cost_vec)
            ofs << x.first << "\t" << x.second << '\n';
    }

    return Q;
}

vector<vector<double>> Simba::calcQuantizationValues(double* X, size_t d, int s, int ell, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, bool debug, double* initial_levels, string log_cost) {
    if ((ell == 1) && (s == 2)) {
        vector<vector<double>> res(1);
        vector<double> inner(2);
        inner[0] = X[0];
        inner[1] = X[d - 1];
        res[0] = inner;
        return res;
    }
    this->X = X;
    this->d = d;
    this->s = s;
    this->ell = ell;
    this->iters = iters;
    this->bin_iters = bin_iters;
    this->bin_iters_increase_threshold = bin_iters_increase_threshold;
    this->stopping_threshold = stopping_threshold;
    this->debug = debug;

    return _calcQuantizationValues(initial_levels, log_cost);
}
