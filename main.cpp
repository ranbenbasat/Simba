#include <stdio.h>              
#include <stdlib.h>             
#include <fstream>
#include "defs.h"
#include "ApproxQUIVER.h"
#include "ExactQUIVER.h"
#include <random>
#include "Simba.h"

int main()
{
    {
        default_random_engine generator;
        generator.seed(42);

        int n = 1 << 20;
        int s = 4;
        int m_QUIVER = 1000;
        int M_QUIVER_for_Simba_Initialization = 1000;
        int ell = 4;
        int iters = 10000000;

        bool run_QUIVER = true;
        bool run_ApxQUIVER = true;
        bool run_Simba = true;

        string log_cost_fn = "";
        auto start = chrono::high_resolution_clock::now();
        auto stop = chrono::high_resolution_clock::now();
        bool debug = false;
        bool truncate = false;     // For optimizing the tables for QUIC-FL
        double T_Truncate = 3.097; // Default value for QUIC-FL
        int bin_iters = 2;
        double bin_iters_increase_threshold = .99;
        double stopping_threshold = .999;

        if (truncate) {             // For optimizing the tables for QUIC-FL, this is an offline computation, and we can thus spend more time optimizing
            bin_iters = 4;
            bin_iters_increase_threshold = .999;
            stopping_threshold = .99999;
        }

        normal_distribution<double> distribution(0.0, 1);
        vector<double> unsorted_X(n);
        vector<double> X;
        int idx = -1;
        double norm = 0;

        for (int i = 0; i < n; ++i) {
            double number = distribution(generator);
            norm += number * number;
            if (!truncate || (number < T_Truncate && number > -T_Truncate)) {
                unsorted_X[++idx] = number;
            }
        }
        if (truncate) {
            n = idx + 1;
        }
        unsorted_X.resize(idx + 1);
        X.resize(idx + 1);
        cout << "norm = " << norm << endl;
        start = chrono::high_resolution_clock::now();
        partial_sort_copy(unsorted_X.begin(), unsorted_X.end(), X.begin(), X.end());
        stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "partial_sort_copy time: " << duration.count() / 1000 << " ms" << endl;
        double vnmse;
        if (run_QUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<false, true> aeq(X.data(), (uint32_t)n, nullptr);
            auto Q = aeq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(X, Q, nullptr);

            cout << "QUIVER Q: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << Q[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "QUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_ApxQUIVER) {
            start = chrono::high_resolution_clock::now();
            ApproxQUIVER<false> taq(X.data(), (uint32_t)n, nullptr, m_QUIVER);
            auto Q = taq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(X, Q, nullptr);

            cout << "ApxQUIVER Q: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << Q[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "ApxQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_Simba) {
            start = chrono::high_resolution_clock::now();
            Simba As;
            vector<double> initial_Q;
            if (M_QUIVER_for_Simba_Initialization == -1) {
                ExactQUIVER<false, true> eq(X.data(), (uint32_t)n, nullptr);
                initial_Q = eq.calcQuantizationValues(s);
            }
            else {
                ApproxQUIVER<false> taq(X.data(), (uint32_t)n, nullptr, M_QUIVER_for_Simba_Initialization);
                initial_Q = taq.calcQuantizationValues(s);
                vnmse = sq_vnmse(X, initial_Q, nullptr);
            }
            auto Q = As.calcQuantizationValues(X.data(), X.size(), s, ell, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, debug, initial_Q.data(), log_cost_fn);
            stop = chrono::high_resolution_clock::now();

            std::vector<std::vector<double>> vec_Q; // reformat Q for calc_SR_vNMSE
            for (int i = 0; i < ell; ++i) {
                std::vector<double> v;
                for (int j = 0; j < s; ++j) {
                    v.push_back(Q[i][j]);
                }
                vec_Q.push_back(v);
            }

            if (truncate) {
                vnmse = calc_SR_vNMSE(X, vec_Q, norm);
            }
            else {
                vnmse = calc_SR_vNMSE(X, vec_Q);
            }

            cout << "Simba Q: ";
            cout << "[" << endl;
            for (auto v : vec_Q) {
                cout << "[";
                for (auto u : v) {
                    cout << u << ", ";
                }
                cout << "]" << endl;
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "Simba time: " << duration.count() / 1000 << " ms" << endl;
        }
    }

    // Check for memory leaks: send all reports to STDOUT
    // _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    // _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    // _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
    // _CrtDumpMemoryLeaks();

    return 0;
}