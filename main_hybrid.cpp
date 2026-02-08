#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // 用來計算時間 (Speedup)
#include <thread> // 用來模擬耗時操作

using namespace std;

// ==========================================
// 1. 系統參數與模擬設定
// ==========================================
const int IMG_SIZE = 64;
const int MIN_BLOCK_SIZE = 4;
const double GT_THRESHOLD = 20.0; // Ground Truth 的切分標準

// 統計數據 (用來畫圖表)
struct Stats {
    long long total_time_ns = 0;
    int blocks_checked = 0;
    int splitting_decisions = 0;
    double error_diff = 0.0; // 和 Ground Truth 的差異
};

Stats stats_baseline;
Stats stats_hybrid;

// ==========================================
// 2. 你的 ML 模型 (The Binary Model)
// ==========================================
bool predict_split_ml(double variance, double grad_h, double grad_v) {
    if (variance <= 5141.6001f) {
        if (grad_h <= 62.2188f) {
            if (grad_h <= 18.2188f) {
                // Leaf: Class 0, Confidence 0.80
                return 0;
            } else {
                // Leaf: Class 0, Confidence 1.00
                return 0;
            }
        } else {
            // Leaf: Class 1, Confidence 1.00
            return 1;
        }
    } else {
        if (variance <= 5754.1450f) {
            if (variance <= 5577.6599f) {
                // Leaf: Class 0, Confidence 0.61
                return 0;
            } else {
                // Leaf: Class 1, Confidence 0.85
                return 1;
            }
        } else {
            if (variance <= 8911.4854f) {
                // Leaf: Class 0, Confidence 0.93
                return 0;
            } else {
                // Leaf: Class 1, Confidence 1.00
                return 1;
            }
        }
    }
}

// ==========================================
// 3. 基礎影像處理函數 (同 Phase 1)
// ==========================================
vector<vector<int>> generate_dummy_image(int size) {
    vector<vector<int>> img(size, vector<int>(size));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x < size/2 && y < size/2) img[y][x] = 10;
            else if (x > size/2 && y > size/2) img[y][x] = 240;
            else img[y][x] = (x * y) % 255;
        }
    }
    return img;
}

void compute_features(const vector<vector<int>>& img, int x, int y, int size, 
                      double& var, double& gh, double& gv) {
    double sum = 0, sq_sum = 0;
    gh = 0; gv = 0;
    int count = size * size;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int val = img[y+i][x+j];
            sum += val;
            sq_sum += val * val;
            if (j+1 < size) gh += abs(val - img[y+i][x+j+1]);
            if (i+1 < size) gv += abs(val - img[y+i+1][x+j]);
        }
    }
    double mean = sum / count;
    var = (sq_sum / count) - (mean * mean);
    gh /= count;
    gv /= count;
}

// 模擬真實 Encoder 的 RDO 耗時計算 (只在 Baseline 模式下會被完整懲罰)
void simulate_heavy_rdo() {
    // 在真實世界，這可能是 DCT + Quantization + Entropy Coding 估算
    // 這裡我們用一個空迴圈模擬 CPU 負載
    volatile int dummy = 0;
    for(int i=0; i<1000; i++) dummy++; 
}

// ==========================================
// 4. 核心邏輯對決：Baseline vs Hybrid
// ==========================================

// --- A. The Heavy C Model (Baseline) ---
void baseline_partition(const vector<vector<int>>& img, int x, int y, int size) {
    auto start = chrono::high_resolution_clock::now();

    // 1. 完整計算 (Heavy)
    simulate_heavy_rdo(); 
    double var, gh, gv;
    compute_features(img, x, y, size, var, gh, gv);
    
    stats_baseline.blocks_checked++;

    // 2. 決策
    bool should_split = (var > GT_THRESHOLD) && (size > MIN_BLOCK_SIZE);

    if (should_split) {
        stats_baseline.splitting_decisions++;
        int half = size / 2;
        baseline_partition(img, x, y, half);
        baseline_partition(img, x + half, y, half);
        baseline_partition(img, x, y + half, half);
        baseline_partition(img, x + half, y + half, half);
    }
    
    auto end = chrono::high_resolution_clock::now();
    stats_baseline.total_time_ns += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
}

// --- B. The Hybrid Model (ML Accelerated) ---
void hybrid_partition(const vector<vector<int>>& img, int x, int y, int size, bool use_ml) {
    auto start = chrono::high_resolution_clock::now();

    // 1. 特徵計算 (假設這部分比 RDO 快很多)
    double var, gh, gv;
    compute_features(img, x, y, size, var, gh, gv);
    stats_hybrid.blocks_checked++;

    bool should_split = false;

    if (use_ml) {
        // [關鍵差異]：直接問 ML，跳過 Heavy RDO
        // 在真實場景，這裡我們假設 compute_features 是輕量的，
        // 而 baseline 裡的 simulate_heavy_rdo() 被我們避開了。
        should_split = predict_split_ml(var, gh, gv); 
    } else {
        // Fallback to rules (如果需要的話)
        should_split = (var > GT_THRESHOLD); 
    }

    // 限制條件
    if (size <= MIN_BLOCK_SIZE) should_split = false;

    if (should_split) {
        stats_hybrid.splitting_decisions++;
        int half = size / 2;
        hybrid_partition(img, x, y, half, use_ml);
        hybrid_partition(img, x + half, y, half, use_ml);
        hybrid_partition(img, x, y + half, half, use_ml);
        hybrid_partition(img, x + half, y + half, half, use_ml);
    }
    
    auto end = chrono::high_resolution_clock::now();
    stats_hybrid.total_time_ns += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
}

// ==========================================
// 5. 主程式：執行並產出報表
// ==========================================
int main() {
    auto img = generate_dummy_image(IMG_SIZE);

    cout << "=== Phase 3: Integration & Performance Benchmark ===\n";
    cout << "Image Size: " << IMG_SIZE << "x" << IMG_SIZE << "\n\n";

    // 1. 跑 Baseline
    cout << "Running Baseline C Model (Full RDO)... ";
    baseline_partition(img, 0, 0, IMG_SIZE);
    cout << "Done.\n";

    // 2. 跑 Hybrid ML Model
    cout << "Running Hybrid ML Model (Fast Inference)... ";
    hybrid_partition(img, 0, 0, IMG_SIZE, true);
    cout << "Done.\n\n";

    // 3. 計算結果
    double time_baseline = stats_baseline.total_time_ns / 1000.0; // microseconds
    double time_hybrid = stats_hybrid.total_time_ns / 1000.0;     // microseconds
    double speedup = time_baseline / time_hybrid;

    // 計算準確度差異 (簡單比較切分次數，真實場景會比對像素)
    // 如果 ML 切分次數和 Baseline 很接近，代表結構預測很準
    int diff = abs(stats_baseline.splitting_decisions - stats_hybrid.splitting_decisions);
    double accuracy_proxy = 100.0 * (1.0 - (double)diff / stats_baseline.splitting_decisions);

    cout << "--------------------------------------------------\n";
    cout << "RESULTS:\n";
    cout << "Baseline Time : " << time_baseline << " us\n";
    cout << "Hybrid Time   : " << time_hybrid << " us\n";
    cout << ">> SPEEDUP    : " << speedup << "x (倍)\n";
    cout << "--------------------------------------------------\n";
    cout << "Baseline Splits: " << stats_baseline.splitting_decisions << "\n";
    cout << "Hybrid Splits  : " << stats_hybrid.splitting_decisions << "\n";
    cout << ">> ACCURACY    : ~" << accuracy_proxy << "% (Structure Match)\n";
    cout << "--------------------------------------------------\n";

    return 0;
}