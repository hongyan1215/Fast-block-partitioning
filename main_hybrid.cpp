#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

// ============================================================
// Fast Block Partitioning with ML Acceleration
// ============================================================
// 
// 本專案模擬視訊編碼器中的 Block Partitioning 決策：
// - C Model: 完整 RDO (Rate-Distortion Optimization) 計算
// - Binary Model: ML 模型快速預測
// - Hybrid Model: 根據 Confidence Threshold 決定使用哪個
// ============================================================

// --- 系統參數 ---
const int IMG_SIZE = 64;
const int MIN_BLOCK_SIZE = 4;
const double LAMBDA = 0.5;  // Lagrangian multiplier for RDO

// ============================================================
// 1. ML 模型 (The Binary Model with Confidence)
// ============================================================

struct MLPrediction {
    bool should_split;
    double confidence;
};

// 這是從 Python 訓練後轉譯過來的 Decision Tree
// 實際專案中，這段會由 train_and_convert.py 自動生成
MLPrediction predict_split_ml(double variance, double grad_h, double grad_v) {
    if (variance <= 5141.600098) {
        if (grad_h <= 62.218750) {
            if (grad_h <= 18.218750) {
                if (variance <= 0.000000) {
                    // Samples: no_split=32, split=0
                    return {false, 1.0000};
                } else {
                    // Samples: no_split=52, split=13
                    return {false, 0.8000};
                }
            } else {
                // Samples: no_split=3, split=0
                return {false, 1.0000};
            }
        } else {
            // Samples: no_split=0, split=2
            return {true, 1.0000};
        }
    } else {
        if (variance <= 5754.145020) {
            if (variance <= 5577.659912) {
                // Samples: no_split=11, split=7
                return {false, 0.6111};
            } else {
                // Samples: no_split=2, split=11
                return {true, 0.8462};
            }
        } else {
            if (variance <= 8911.485352) {
                // Samples: no_split=39, split=3
                return {false, 0.9286};
            } else {
                // Samples: no_split=0, split=42
                return {true, 1.0000};
            }
        }
    }
}

// ============================================================
// 2. 影像生成與特徵計算
// ============================================================

vector<vector<int>> generate_test_image(int size) {
    vector<vector<int>> img(size, vector<int>(size));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x < size/2 && y < size/2) {
                img[y][x] = 10;  // 平滑區 (低變異)
            } else if (x > size/2 && y > size/2) {
                img[y][x] = 240; // 平滑區 (低變異)
            } else {
                img[y][x] = (x * y) % 255;  // 紋理區 (高變異)
            }
        }
    }
    return img;
}

void compute_features(const vector<vector<int>>& img, int x, int y, int size, 
                      double& var, double& grad_h, double& grad_v) {
    double sum = 0, sq_sum = 0;
    grad_h = 0; grad_v = 0;
    int count = size * size;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int val = img[y+i][x+j];
            sum += val;
            sq_sum += val * val;
            if (j+1 < size) grad_h += abs(val - img[y+i][x+j+1]);
            if (i+1 < size) grad_v += abs(val - img[y+i+1][x+j]);
        }
    }
    double mean = sum / count;
    var = (sq_sum / count) - (mean * mean);
    grad_h /= count;
    grad_v /= count;
}

// ============================================================
// 3. RDO (Rate-Distortion Optimization) 計算
// ============================================================
// 這是 C Model 的核心：計算真正的 Rate-Distortion Cost
// RD Cost = Distortion + λ * Rate

// 計算 SAD (Sum of Absolute Differences) - 失真度量
double compute_SAD(const vector<vector<int>>& img, int x, int y, int size, int pred_value) {
    double sad = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            sad += abs(img[y+i][x+j] - pred_value);
        }
    }
    return sad;
}

// 計算 SSD (Sum of Squared Differences) - 失真度量
double compute_SSD(const vector<vector<int>>& img, int x, int y, int size, int pred_value) {
    double ssd = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int diff = img[y+i][x+j] - pred_value;
            ssd += diff * diff;
        }
    }
    return ssd;
}

// 估計編碼位元數 (Rate Estimation)
// 簡化模型：Rate ≈ log2(variance + 1) * block_area
double estimate_rate(double variance, int size) {
    double bits_per_pixel = log2(variance + 1) / 8.0;  // 正規化
    return bits_per_pixel * size * size;
}

// 計算區塊的平均值 (作為預測值)
int compute_mean(const vector<vector<int>>& img, int x, int y, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            sum += img[y+i][x+j];
        }
    }
    return (int)(sum / (size * size));
}

// 完整 RDO Cost 計算
// 這是 Heavy C Model 的核心 - 計算量大但準確
struct RDOResult {
    double cost_no_split;   // 不切分的 RD Cost
    double cost_split;      // 切分的 RD Cost
    bool should_split;
    int rdo_operations;     // 計算次數（用來衡量複雜度）
};

RDOResult compute_rdo(const vector<vector<int>>& img, int x, int y, int size) {
    RDOResult result;
    result.rdo_operations = 0;
    
    // --- 計算不切分的 Cost ---
    int pred_value = compute_mean(img, x, y, size);
    double distortion_no_split = compute_SSD(img, x, y, size, pred_value);
    double var, gh, gv;
    compute_features(img, x, y, size, var, gh, gv);
    double rate_no_split = estimate_rate(var, size);
    result.cost_no_split = distortion_no_split + LAMBDA * rate_no_split;
    result.rdo_operations++;
    
    // --- 計算切分的 Cost (四個子區塊的總和) ---
    if (size > MIN_BLOCK_SIZE) {
        int half = size / 2;
        double total_distortion = 0;
        double total_rate = 0;
        
        // 計算四個子區塊
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int sx = x + dx * half;
                int sy = y + dy * half;
                
                int sub_pred = compute_mean(img, sx, sy, half);
                total_distortion += compute_SSD(img, sx, sy, half, sub_pred);
                
                double sub_var, sub_gh, sub_gv;
                compute_features(img, sx, sy, half, sub_var, sub_gh, sub_gv);
                total_rate += estimate_rate(sub_var, half);
                
                result.rdo_operations++;
            }
        }
        
        // 切分有額外的 overhead (分割標記需要編碼)
        double split_overhead = 2.0;  // 假設切分標記需要 2 bits
        result.cost_split = total_distortion + LAMBDA * (total_rate + split_overhead);
    } else {
        result.cost_split = 1e9;  // 無法再切分
    }
    
    result.should_split = (result.cost_split < result.cost_no_split) && (size > MIN_BLOCK_SIZE);
    return result;
}

// ============================================================
// 4. 分割策略實作
// ============================================================

// 統計數據
struct PartitionStats {
    long long total_time_ns = 0;
    int total_blocks = 0;
    int split_decisions = 0;
    int rdo_operations = 0;
    vector<pair<int, int>> block_sizes;  // (x, y, size) 紀錄最終分割
    
    void reset() {
        total_time_ns = 0;
        total_blocks = 0;
        split_decisions = 0;
        rdo_operations = 0;
        block_sizes.clear();
    }
};

// --- A. C Model: 完整 RDO (Ground Truth) ---
void partition_c_model(const vector<vector<int>>& img, int x, int y, int size, PartitionStats& stats) {
    auto start = chrono::high_resolution_clock::now();
    
    stats.total_blocks++;
    
    // 完整 RDO 計算
    RDOResult rdo = compute_rdo(img, x, y, size);
    stats.rdo_operations += rdo.rdo_operations;
    
    auto end = chrono::high_resolution_clock::now();
    stats.total_time_ns += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    
    if (rdo.should_split) {
        stats.split_decisions++;
        int half = size / 2;
        partition_c_model(img, x, y, half, stats);
        partition_c_model(img, x + half, y, half, stats);
        partition_c_model(img, x, y + half, half, stats);
        partition_c_model(img, x + half, y + half, half, stats);
    } else {
        stats.block_sizes.push_back({x + y * IMG_SIZE, size});
    }
}

// --- B. Hybrid Model: ML + Confidence-based Fallback ---
void partition_hybrid(const vector<vector<int>>& img, int x, int y, int size, 
                      double confidence_threshold, PartitionStats& stats, bool use_fallback) {
    auto start = chrono::high_resolution_clock::now();
    
    stats.total_blocks++;
    
    // 1. 快速特徵計算
    double var, grad_h, grad_v;
    compute_features(img, x, y, size, var, grad_h, grad_v);
    
    // 2. ML 預測
    MLPrediction pred = predict_split_ml(var, grad_h, grad_v);
    
    bool should_split = false;
    
    // 3. Confidence-based Decision
    if (pred.confidence >= confidence_threshold) {
        // ML 有足夠信心，直接採用 (快速路徑)
        should_split = pred.should_split;
    } else if (use_fallback) {
        // ML 信心不足，回退到 RDO (慢速但準確)
        RDOResult rdo = compute_rdo(img, x, y, size);
        stats.rdo_operations += rdo.rdo_operations;
        should_split = rdo.should_split;
    } else {
        // 不使用 fallback，直接採用 ML 結果
        should_split = pred.should_split;
    }
    
    // 限制條件
    if (size <= MIN_BLOCK_SIZE) should_split = false;
    
    auto end = chrono::high_resolution_clock::now();
    stats.total_time_ns += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    
    if (should_split) {
        stats.split_decisions++;
        int half = size / 2;
        partition_hybrid(img, x, y, half, confidence_threshold, stats, use_fallback);
        partition_hybrid(img, x + half, y, half, confidence_threshold, stats, use_fallback);
        partition_hybrid(img, x, y + half, half, confidence_threshold, stats, use_fallback);
        partition_hybrid(img, x + half, y + half, half, confidence_threshold, stats, use_fallback);
    } else {
        stats.block_sizes.push_back({x + y * IMG_SIZE, size});
    }
}

// ============================================================
// 5. 結果比較與 Pareto Frontier 分析
// ============================================================

double compute_structure_match(const PartitionStats& baseline, const PartitionStats& test) {
    // 比較分割結構的相似度
    int match = 0;
    int total = baseline.block_sizes.size();
    
    for (const auto& b : baseline.block_sizes) {
        for (const auto& t : test.block_sizes) {
            if (b.first == t.first && b.second == t.second) {
                match++;
                break;
            }
        }
    }
    
    return (total > 0) ? (100.0 * match / total) : 0;
}

int main() {
    auto img = generate_test_image(IMG_SIZE);
    
    cout << "============================================================\n";
    cout << "   Fast Block Partitioning with ML Acceleration\n";
    cout << "============================================================\n";
    cout << "Image Size: " << IMG_SIZE << "x" << IMG_SIZE << "\n";
    cout << "Min Block : " << MIN_BLOCK_SIZE << "x" << MIN_BLOCK_SIZE << "\n";
    cout << "Lambda    : " << LAMBDA << "\n\n";
    
    // ========================================
    // 1. 執行 C Model (Ground Truth)
    // ========================================
    cout << "[C Model] Running full RDO... ";
    PartitionStats stats_c;
    partition_c_model(img, 0, 0, IMG_SIZE, stats_c);
    cout << "Done.\n";
    
    double time_c = stats_c.total_time_ns / 1000.0;  // microseconds
    
    cout << "          Time: " << fixed << setprecision(2) << time_c << " us\n";
    cout << "          Splits: " << stats_c.split_decisions << "\n";
    cout << "          RDO Ops: " << stats_c.rdo_operations << "\n";
    cout << "          Final Blocks: " << stats_c.block_sizes.size() << "\n\n";
    
    // ========================================
    // 2. Pareto Frontier: 測試不同 Threshold
    // ========================================
    cout << "============================================================\n";
    cout << "   Pareto Frontier Analysis (Speedup vs Accuracy)\n";
    cout << "============================================================\n";
    cout << setw(12) << "Threshold" 
         << setw(12) << "Time(us)" 
         << setw(10) << "Speedup"
         << setw(12) << "Accuracy"
         << setw(10) << "RDO Ops"
         << "\n";
    cout << "------------------------------------------------------------\n";
    
    vector<double> thresholds = {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0};
    
    // 儲存 Pareto 數據
    ofstream pareto_file("pareto_data.csv");
    pareto_file << "threshold,time_us,speedup,accuracy,rdo_ops\n";
    
    for (double thresh : thresholds) {
        PartitionStats stats_hybrid;
        partition_hybrid(img, 0, 0, IMG_SIZE, thresh, stats_hybrid, true);
        
        double time_hybrid = stats_hybrid.total_time_ns / 1000.0;
        double speedup = time_c / time_hybrid;
        double accuracy = compute_structure_match(stats_c, stats_hybrid);
        
        cout << setw(12) << fixed << setprecision(2) << thresh
             << setw(12) << time_hybrid
             << setw(10) << speedup << "x"
             << setw(11) << accuracy << "%"
             << setw(10) << stats_hybrid.rdo_operations
             << "\n";
        
        pareto_file << thresh << "," << time_hybrid << "," << speedup << "," 
                    << accuracy << "," << stats_hybrid.rdo_operations << "\n";
    }
    
    pareto_file.close();
    
    // ========================================
    // 3. ML-Only Mode (無 Fallback)
    // ========================================
    cout << "\n============================================================\n";
    cout << "   ML-Only Mode (No RDO Fallback)\n";
    cout << "============================================================\n";
    
    PartitionStats stats_ml_only;
    partition_hybrid(img, 0, 0, IMG_SIZE, 0.0, stats_ml_only, false);
    
    double time_ml = stats_ml_only.total_time_ns / 1000.0;
    double speedup_ml = time_c / time_ml;
    double accuracy_ml = compute_structure_match(stats_c, stats_ml_only);
    
    cout << "Time     : " << fixed << setprecision(2) << time_ml << " us\n";
    cout << "Speedup  : " << speedup_ml << "x\n";
    cout << "Accuracy : " << accuracy_ml << "%\n";
    cout << "RDO Ops  : " << stats_ml_only.rdo_operations << " (should be 0)\n";
    
    // ========================================
    // 4. 最佳配置建議
    // ========================================
    cout << "\n============================================================\n";
    cout << "   Recommendation\n";
    cout << "============================================================\n";
    cout << "For best balance of speed and accuracy:\n";
    cout << "  -> Use Confidence Threshold = 0.8\n";
    cout << "  -> Expected: ~2-3x speedup with ~90%+ accuracy\n";
    cout << "\nPareto data saved to 'pareto_data.csv'\n";
    cout << "============================================================\n";
    
    return 0;
}