#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct TrialSample {
    int label;
    std::vector<double> features;
};

static constexpr int kNumClasses = 4;
static constexpr int kFeatDim = 12;

std::vector<std::vector<double>> build_source_centroids() {
    std::vector<std::vector<double>> centroids(kNumClasses, std::vector<double>(kFeatDim, 0.0));
    for (int c = 0; c < kNumClasses; ++c) {
        for (int d = 0; d < kFeatDim; ++d) {
            double angle = 0.55 * static_cast<double>(c + 1) * static_cast<double>(d + 1);
            centroids[c][d] = 1.15 * std::sin(angle) + 0.55 * std::cos(0.35 * c - 0.4 * d);
        }
    }
    return centroids;
}

std::vector<double> transform_target_point(
    const std::vector<double>& src_vec,
    double shift_scale
) {
    std::vector<double> out(kFeatDim, 0.0);
    double theta = 0.55 * shift_scale;
    double c = std::cos(theta);
    double s = std::sin(theta);

    for (int d = 0; d < kFeatDim; d += 2) {
        double x0 = src_vec[d];
        double x1 = src_vec[d + 1];
        double gain0 = 1.0 + 0.30 * shift_scale * std::sin(0.5 * d + 0.3);
        double gain1 = 1.0 - 0.25 * shift_scale * std::cos(0.4 * d + 0.2);
        double bias0 = 0.45 * shift_scale * std::cos(0.6 * d);
        double bias1 = -0.40 * shift_scale * std::sin(0.5 * d);

        out[d] = gain0 * (c * x0 - s * x1) + bias0;
        out[d + 1] = gain1 * (s * x0 + c * x1) + bias1;
    }
    return out;
}

std::vector<TrialSample> generate_split(
    int n_trials,
    double shift_scale,
    double noise_std,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, noise_std);
    auto base_centroids = build_source_centroids();

    std::vector<TrialSample> data;
    data.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        int cls = i % kNumClasses;
        std::vector<double> raw = base_centroids[cls];
        for (int d = 0; d < kFeatDim; ++d) {
            raw[d] += dist(rng);
        }
        std::vector<double> shifted = (shift_scale > 0.0)
            ? transform_target_point(raw, shift_scale)
            : raw;
        data.push_back(TrialSample{cls, shifted});
    }
    return data;
}

std::vector<std::vector<double>> estimate_centroids(const std::vector<TrialSample>& data) {
    std::vector<std::vector<double>> sums(kNumClasses, std::vector<double>(kFeatDim, 0.0));
    std::vector<int> counts(kNumClasses, 0);

    for (const auto& item : data) {
        counts[item.label] += 1;
        for (int d = 0; d < kFeatDim; ++d) {
            sums[item.label][d] += item.features[d];
        }
    }

    for (int c = 0; c < kNumClasses; ++c) {
        int denom = std::max(1, counts[c]);
        for (int d = 0; d < kFeatDim; ++d) {
            sums[c][d] /= static_cast<double>(denom);
        }
    }
    return sums;
}

std::vector<double> compute_global_mean(const std::vector<TrialSample>& data) {
    std::vector<double> mean(kFeatDim, 0.0);
    if (data.empty()) return mean;
    for (const auto& item : data) {
        for (int d = 0; d < kFeatDim; ++d) {
            mean[d] += item.features[d];
        }
    }
    double denom = static_cast<double>(data.size());
    for (int d = 0; d < kFeatDim; ++d) {
        mean[d] /= denom;
    }
    return mean;
}

std::vector<double> compute_global_std(
    const std::vector<TrialSample>& data,
    const std::vector<double>& mean
) {
    std::vector<double> stdev(kFeatDim, 1.0);
    if (data.size() < 2) return stdev;
    std::vector<double> var(kFeatDim, 0.0);
    for (const auto& item : data) {
        for (int d = 0; d < kFeatDim; ++d) {
            double diff = item.features[d] - mean[d];
            var[d] += diff * diff;
        }
    }
    double denom = static_cast<double>(data.size() - 1);
    for (int d = 0; d < kFeatDim; ++d) {
        stdev[d] = std::sqrt(var[d] / denom + 1e-6);
    }
    return stdev;
}

std::vector<double> zscore_vec(
    const std::vector<double>& x,
    const std::vector<double>& mean,
    const std::vector<double>& stdev
) {
    std::vector<double> out(kFeatDim, 0.0);
    for (int d = 0; d < kFeatDim; ++d) {
        out[d] = (x[d] - mean[d]) / stdev[d];
    }
    return out;
}

int predict_nearest_centroid(
    const std::vector<double>& feat,
    const std::vector<std::vector<double>>& centroids
) {
    int best_cls = 0;
    double best_dist = 1e18;
    for (int c = 0; c < kNumClasses; ++c) {
        double dist_sq = 0.0;
        for (int d = 0; d < kFeatDim; ++d) {
            double diff = feat[d] - centroids[c][d];
            dist_sq += diff * diff;
        }
        if (dist_sq < best_dist) {
            best_dist = dist_sq;
            best_cls = c;
        }
    }
    return best_cls;
}

int main(int argc, char* argv[]) {
    double shift_scale = (argc >= 2) ? std::atof(argv[1]) : 1.00;
    int n_trials = (argc >= 3) ? std::max(40, std::atoi(argv[2])) : 320;
    double noise_std = (argc >= 4) ? std::atof(argv[3]) : 0.55;
    std::string out_csv = (argc >= 5) ? argv[4] : "";

    auto src_train = generate_split(n_trials, 0.0, noise_std, 101);
    auto tgt_all = generate_split(n_trials, shift_scale, noise_std, 202);

    int adapt_count = n_trials / 4;
    std::vector<TrialSample> tgt_adapt(tgt_all.begin(), tgt_all.begin() + adapt_count);
    std::vector<TrialSample> tgt_test(tgt_all.begin() + adapt_count, tgt_all.end());

    // Zero-shot decoder: source centroids evaluated directly on target test trials
    auto src_centroids = estimate_centroids(src_train);

    // Adapted decoder: target feature alignment (z-score matching) + source-regularized target prototypes
    auto src_mean = compute_global_mean(src_train);
    auto src_std = compute_global_std(src_train, src_mean);
    auto tgt_mean = compute_global_mean(tgt_adapt);
    auto tgt_std = compute_global_std(tgt_adapt, tgt_mean);

    std::vector<TrialSample> src_norm = src_train;
    for (auto& item : src_norm) {
        item.features = zscore_vec(item.features, src_mean, src_std);
    }
    std::vector<TrialSample> adapt_norm = tgt_adapt;
    for (auto& item : adapt_norm) {
        item.features = zscore_vec(item.features, tgt_mean, tgt_std);
    }

    auto norm_src_centroids = estimate_centroids(src_norm);
    auto norm_tgt_centroids = estimate_centroids(adapt_norm);
    std::vector<std::vector<double>> adapted_centroids(
        kNumClasses, std::vector<double>(kFeatDim, 0.0)
    );
    for (int c = 0; c < kNumClasses; ++c) {
        for (int d = 0; d < kFeatDim; ++d) {
            adapted_centroids[c][d] = 0.35 * norm_src_centroids[c][d] + 0.65 * norm_tgt_centroids[c][d];
        }
    }

    int zs_correct = 0;
    int ad_correct = 0;

    std::ofstream fout;
    if (!out_csv.empty()) {
        fout.open(out_csv);
        fout << "Trial,True_Label,ZeroShot_Pred,Adapted_Pred,ZeroShot_Correct,Adapted_Correct\n";
    }

    for (size_t i = 0; i < tgt_test.size(); ++i) {
        const auto& sample = tgt_test[i];
        int zs_pred = predict_nearest_centroid(sample.features, src_centroids);
        auto aligned_feat = zscore_vec(sample.features, tgt_mean, tgt_std);
        int ad_pred = predict_nearest_centroid(aligned_feat, adapted_centroids);

        int zs_ok = (zs_pred == sample.label) ? 1 : 0;
        int ad_ok = (ad_pred == sample.label) ? 1 : 0;
        zs_correct += zs_ok;
        ad_correct += ad_ok;

        if (fout.is_open()) {
            fout << i << ","
                 << sample.label << ","
                 << zs_pred << ","
                 << ad_pred << ","
                 << zs_ok << ","
                 << ad_ok << "\n";
        }
    }

    double denom = static_cast<double>(tgt_test.size());
    double zs_acc = static_cast<double>(zs_correct) / denom;
    double ad_acc = static_cast<double>(ad_correct) / denom;
    double gain = ad_acc - zs_acc;

    std::cout << std::fixed << std::setprecision(4)
              << "Shift_Scale=" << std::setprecision(3) << shift_scale
              << std::setprecision(4)
              << " ZeroShot_Acc=" << zs_acc
              << " Adapted_Acc=" << ad_acc
              << " Transfer_Gain=" << (gain >= 0.0 ? "+" : "") << gain << "\n";

    return 0;
}
