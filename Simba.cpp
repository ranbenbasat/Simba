#include "Simba.h"

/*
* Calculate the thresholds based on the current levels. If changed_H and changed_X are -1, calculate all thresholds from scratch, otherwise only update the thresholds affected by the change in levels[changed_H][changed_X].
*/
void Simba::calcThresholds(int changed_H, int changed_X)
{
    if (changed_X == -1)
    {
        double T = 0;
        for (int j = 1; j < ell; ++j) {
            T += levels[j][0];
        }
        T += levels[0][1];
        //new_threshold_indices[0] = upper_bound(svec, svec + d, T / ell) - svec - 1;
        new_threshold_indices[0] = upper_bound(svec, svec + d, T * 1.0000000001 / ell) - svec - 1; // multiplying by (1+eps) to avoid numerical errors. Don't remove!
        threshold_values[0] = T / ell;

        int idx = 0;
        for (int H = 1; H < ell; ++H) {
            T += levels[H][1] - levels[H][0];
            new_threshold_indices[++idx] = (upper_bound(svec, svec + d, T * 1.0000000001 / ell) - svec) - 1;
            threshold_values[idx] = T / ell;
        }
        for (int X = 1; X < s - 1; ++X) {
            for (int H = 0; H < ell; ++H) {
                T += levels[H][X + 1] - levels[H][X];
                new_threshold_indices[++idx] = (upper_bound(svec, svec + d, T * 1.0000000001 / ell) - svec) - 1;
                threshold_values[idx] = T / ell;
            }
        }
        new_threshold_indices[ell * (s - 1) - 1] = d - 1; //The last threshold should include the end of the vector, this avoids numerical issues.
    }
    else
    {
        int changed_j = changed_H + changed_X * ell;
        int first_changed_i = changed_j > ell ? changed_j - (ell - 1) : 1;
        double T = 0;
        for (int j = first_changed_i; j < first_changed_i + ell; ++j) {
            T += levels[j % ell][j / ell];
        }
        new_threshold_indices[first_changed_i - 1] = (upper_bound(svec, svec + d, T * 1.0000000001 / ell) - svec) - 1;
        new_threshold_indices[first_changed_i - 1] = new_threshold_indices[first_changed_i - 1] < 0 ? 0 : new_threshold_indices[first_changed_i - 1];
        threshold_values[first_changed_i - 1] = T / ell;
        int last_i = changed_j < ell * (s - 1) - 1 ? changed_j : ell * (s - 1) - 1;
        for (int i = first_changed_i + 1; i <= last_i; ++i) {
            int H = (i - 1) % ell;
            int X = (i - 1) / ell;
            T += levels[H][X + 1] - levels[H][X];
            new_threshold_indices[i - 1] = (upper_bound(svec, svec + d, T * 1.0000000001 / ell) - svec) - 1;
            threshold_values[i - 1] = T / ell;
        }
        new_threshold_indices[ell * (s - 1) - 1] = d - 1;
        threshold_values[ell * (s - 1) - 1] = maxD;
    }
    return;
}


void Simba::calcBetas() {
    int N = new_threshold_indices.size();
    if (N == 0) return;

    // compute betai[0] from scratch
    double sum = 0;
    for (int j = 0; j < ell; ++j) {
        int idx = (ell - j) / ell;  // same as (0 + ell - j) / ell
        double v = levels[j][idx];
        sum += v * v;
    }
    betai[0] = sum;

    // roll-forward: only j0 = (i+1)%ell changes its index
    for (int i = 0; i + 1 < N; ++i) {
        int j0 = (i + 1) % ell;
        int old_idx = (i + ell - j0) / ell;
        int new_idx = (i + 1 + ell - j0) / ell;
        double old_v = levels[j0][old_idx];
        double new_v = levels[j0][new_idx];
        betai[i + 1] = betai[i] + new_v * new_v - old_v * old_v;
    }
}

void Simba::updateBetas(int j, int q, double oldval, double newval) {
    int N = new_threshold_indices.size();
    // change in the squared term
    double diff = newval * newval - oldval * oldval;

    // solve (i + ell - j) / ell == q  =>  (q-1)*ell + j <= i < q*ell + j
    int i_start = (q - 1) * ell + j;
    int i_end = q * ell + j;

    // clamp to valid range
    if (i_start < 0)    i_start = 0;
    if (i_end > N)    i_end = N;

    // update only those betai[i] that include levels[j][q]^2
    for (int i = i_start; i < i_end; ++i) {
        betai[i] += diff;
    }
}


double Simba::calc_MSEi(int i) const {
    int j_star = i % ell;
    int R_idx = (i + ell - j_star) / ell;
    int L_idx = R_idx - 1;
    double Ri = levels[j_star][R_idx];
    double Li = levels[j_star][L_idx];
    double Ti = threshold_values[i];
    double Ni = (i == 0) ? new_threshold_indices[i] + 1 : new_threshold_indices[i] - new_threshold_indices[i - 1];
    double betaival = betai[i];
    double pcumsum = (i == 0) ? pts_cumsum[new_threshold_indices[i]] : pts_cumsum[new_threshold_indices[i]] - pts_cumsum[new_threshold_indices[i - 1]];
    return (Ri + Li) * pcumsum + (betaival / ell - Ti * (Ri + Li)) * Ni;
}

double Simba::cost(int changed_H = -1, int changed_X = -1) {
    calcThresholds(changed_H, changed_X);
    if (changed_X == -1) {
        double total_cost = 0;
        calcBetas();
        for (int i = 0; i < new_threshold_indices.size(); ++i) {
            MSEi[i] = calc_MSEi(i);
            total_cost += MSEi[i];
        }
        lastMSE = total_cost - sos;
    }
    else {
        int level_idx = changed_H + changed_X * ell;
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
    minD = svec[0];
    maxD = svec[d - 1];
    pts_cumsum.resize(d);
    pts_cumsum[0] = minD;
    //partial_sum(svec.begin(), svec.end(), pts_cumsum.begin()); is actually slower ?!
    sos = minD * minD;
    //auto start = chrono::high_resolution_clock::now();
    for (int i = 1; i < d; ++i) {
        pts_cumsum[i] = pts_cumsum[i - 1] + svec[i];
        sos += svec[i] * svec[i];
    }
    //auto end = chrono::high_resolution_clock::now();
    //cout << "Preprocessing took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms." << endl;
}




//////////////////////////////////



vector<vector<double>> Simba::calcQuantizationValuesSR(double* initial_levels, string log_cost)
{
    if (log_cost != "") {
        cout << "WARNING: Calculating quantization values with log_cost enabled." << endl;
    }
    new_threshold_indices.resize(ell * (s - 1));
    threshold_values = new double[ell * (s - 1)] {};
    MSEi = new double[ell * (s - 1)] {};
    betai = new double[ell * (s - 1)] {};

    preprocess();

    levels.resize(ell);
    if (initial_levels) {
        for (int i = 0; i < ell; ++i) {
            levels[i].resize(s);
            for (int j = 0; j < s; ++j)
                levels[i][j] = initial_levels[j];
        }
    }
    else {
        vector<double> lin(s * ell);
        for (int i = 0; i < s * ell - 2; ++i) {
            lin[i + 1] = svec[(int)(((i + .5) * d) / (s * ell - 2))];
        }
        double first = ell * minD;
        for (int i = 0; i < ell - 1; ++i) {
            first -= lin[i + 1];
        }
        lin[0] = first;
        double last = ell * maxD;
        for (int i = 0; i < ell - 1; ++i) {
            last -= lin[i + ell];
        }
        lin[s * ell - 1] = last;
        for (int i = 0; i < ell; ++i) {
            levels[i].resize(s);
            for (int j = 0; j < s; ++j)
                levels[i][j] = lin[i + j * ell];
        }
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


    //double orig_cost = cost();
    //min_required_improvement = orig_cost * (1 - stopping_threshold);
    _cost = cost();
    min_required_improvement = _cost * (1 - stopping_threshold);
    vector<pair<double, int>> cost_vec;


    double lo, hi, mid, mid_plus_eps, cost_mid, cost_mid_plus_eps;
    int iter_num = 0;
    int H = 1, X = 0;
    bool last_changed_was_hi;
    for (; iter_num < iters; ++iter_num) {
        int remaining_bin_iters = bin_iters;
        _cost = cost(H, X);
        H = changed_levels[iter_num % (s * ell - 2)].first;
        X = changed_levels[iter_num % (s * ell - 2)].second;


        if ((log_cost != "") && (iter_num % 1 == 0)) {
            cost_vec.push_back(make_pair(_cost, totally_stuck));
        }
        double orig_val = levels[H][X];

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

        if (H == 0) {
            lo = levels[ell - 1][X - 1];
        }
        else {
            lo = levels[H - 1][X];
            if (X == 0) {
                lo = max(lo, levels[H][X] - (levels[1][0] - levels[0][0]));
                if (H == 1) {
                    lo = max(lo, (levels[1][0] + levels[0][0]) / 2);
                }
            }
        }
        if (H == ell - 1) {
            hi = levels[0][X + 1];
        }
        else {
            hi = levels[H + 1][X];
            if (X == s - 1) {
                hi = min(hi, levels[H][X] + (levels[ell - 1][s - 1] - levels[ell - 2][s - 1]));
                if (H == ell - 2) {
                    hi = min(hi, (levels[ell - 1][s - 1] + levels[ell - 2][s - 1]) / 2);
                }
            }
        }
        if (lo >= hi - 1e-7)
            continue;

        mid = (hi + lo) / 2;
        mid_plus_eps = mid + (hi - mid) * 1e-5;
        while (--remaining_bin_iters >= 0) {
            updateBetas(H, X, levels[H][X], mid);
            levels[H][X] = mid;
            if (X == 0) {
                double res = ell * minD;
                for (int i = 1; i < ell; ++i) {
                    res -= levels[i][0];
                }
                updateBetas(0, 0, levels[0][0], res);
                levels[0][0] = res;
            }
            else if (X == s - 1) {
                double res = ell * maxD;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= levels[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, levels[ell - 1][s - 1], res);
                levels[ell - 1][s - 1] = res;
            }
            cost_mid = cost(H, X);

            updateBetas(H, X, levels[H][X], mid_plus_eps);
            levels[H][X] = mid_plus_eps;
            if (X == 0) {
                // this can be optimized!
                double res = ell * minD;
                for (int i = 1; i < ell; ++i) {
                    res -= levels[i][0];
                }
                updateBetas(0, 0, levels[0][0], res);
                levels[0][0] = res;
            }
            else if (X == s - 1) {
                double res = ell * maxD;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= levels[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, levels[ell - 1][s - 1], res);
                levels[ell - 1][s - 1] = res;
            }
            cost_mid_plus_eps = cost(H, X);
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
        //cost();
        if (last_changed_was_hi)
        {
            updateBetas(H, X, levels[H][X], lo);
            levels[H][X] = lo;
            if (X == 0) {
                // this can be optimized!
                double res = ell * minD;
                for (int i = 1; i < ell; ++i) {
                    res -= levels[i][0];
                }
                updateBetas(0, 0, levels[0][0], res);
                levels[0][0] = res;
            }
            else if (X == s - 1) {
                double res = ell * maxD;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= levels[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, levels[ell - 1][s - 1], res);
                levels[ell - 1][s - 1] = res;
            }
            cost_lo = cost(H, X);
            cost_hi = cost_mid;
        }
        else
        {
            updateBetas(H, X, levels[H][X], hi);
            levels[H][X] = hi;
            if (X == 0) {
                // this can be optimized!
                double res = ell * minD;
                for (int i = 1; i < ell; ++i) {
                    res -= levels[i][0];
                }
                updateBetas(0, 0, levels[0][0], res);
                levels[0][0] = res;
            }
            else if (X == s - 1) {
                double res = ell * maxD;
                for (int i = 0; i < ell - 1; ++i) {
                    res -= levels[i][s - 1];
                }
                updateBetas(ell - 1, s - 1, levels[ell - 1][s - 1], res);
                levels[ell - 1][s - 1] = res;
            }
            cost_hi = cost(H, X);
            cost_lo = cost_mid;
        }

        if ((_cost > cost_lo) || (_cost > cost_hi)) {
            double newval = (cost_lo < cost_hi) ? lo : hi;
            updateBetas(H, X, levels[H][X], newval);
            levels[H][X] = newval;
        }
        else {
            updateBetas(H, X, levels[H][X], orig_val);
            levels[H][X] = orig_val;
        }
        if (X == 0) {
            // this can be optimized!
            double res = ell * minD;
            for (int i = 1; i < ell; ++i) {
                res -= levels[i][0];
            }
            updateBetas(0, 0, levels[0][0], res);
            levels[0][0] = res;
        }
        else if (X == s - 1) {
            // this can be optimized!
            double res = ell * maxD;
            for (int i = 0; i < ell - 1; ++i) {
                res -= levels[i][s - 1];
            }
            updateBetas(ell - 1, s - 1, levels[ell - 1][s - 1], res);
            levels[ell - 1][s - 1] = res;
        }
    }

    if (debug) {
        for (int H = 0; H < ell; ++H) {
            for (int S = 0; S < s - 1; ++S) {
                double res = 0;
                for (int H_prime = H + 1; H_prime < ell; ++H_prime) {
                    res -= levels[H_prime][S];
                }
                for (int H_prime = 0; H_prime < H; ++H_prime) {
                    res -= levels[H_prime][S + 1];
                }
            }
        }

        double norm2 = 0;
        for (int i = 0; i < d; ++i) {
            norm2 += svec[i] * svec[i];
        }

        double  __cost = cost();
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

    return levels;
}


//////
vector<vector<double>> Simba::calcQuantizationValues(double* svec, size_t d, int s, int q, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, int m, bool debug, string quantype, double* initial_levels, string log_cost) {
    if ((q == 1) && (s == 2)) {
        vector<vector<double>> res(1);
        vector<double> inner(2);
        inner[0] = svec[0];
        inner[1] = svec[d - 1];
        res[0] = inner;
        return res;
    }
    this->svec = svec;
    this->d = d;
    this->s = s;
    this->ell = q;
    this->iters = iters;
    this->bin_iters = bin_iters;
    this->bin_iters_increase_threshold = bin_iters_increase_threshold;
    this->stopping_threshold = stopping_threshold;
    this->debug = debug;
    if (m == -1) { // no histogram
        vector<vector<double>> res;

        res = calcQuantizationValuesSR(initial_levels, log_cost);

        return res;
    }

    cout << "m!= -1 is Unimplemented, use OldSimba" << endl;

}
