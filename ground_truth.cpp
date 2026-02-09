#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;

// --- 設定參數 ---
const int IMG_SIZE = 64;       // 圖片大小 (64x64)
const int MIN_BLOCK_SIZE = 4;  // 最小切分單位
const double LAMBDA = 0.5;     // RDO 的 Lagrangian multiplier

// --- 資料結構：用來存給 Python 訓練的資料 ---
struct TrainingData {
    double variance;
    double grad_h; // 水平梯度強度
    double grad_v; // 垂直梯度強度
    int size;
    int should_split; // Label: 1 = 切, 0 = 不切 (由 RDO 決定！)
};

vector<TrainingData> dataset;

// 這裡產生一個簡單的圖案：左上角全黑，右下角全白，中間有漸層 (高頻區)
vector<vector<int>> generate_dummy_image(int size) {
    vector<vector<int>> img(size, vector<int>(size));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x < size / 2 && y < size / 2) {
                img[y][x] = 10; // 平滑區 (低變異)
            } else if (x > size / 2 && y > size / 2) {
                img[y][x] = 240; // 平滑區 (低變異)
            } else {
                // 邊緣/紋理區 (高變異)
                img[y][x] = (x * y) % 255; 
            }
        }
    }
    return img;
}

// --- 核心數學：計算以x,y為左上角，size邊長區塊內的變異數 (Variance) ---
double compute_variance(const vector<vector<int>>& img, int x, int y, int size) {
    double sum = 0.0;
    double sq_sum = 0.0;
    int pixel_count = size * size;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int val = img[y + i][x + j];
            sum += val;
            sq_sum += (val * val);
        }
    }
    
    double mean = sum / pixel_count;
    double variance = (sq_sum / pixel_count) - (mean * mean);
    return variance;
}

// --- 核心數學：計算簡單的梯度 (Gradient) ---
void compute_gradients(const vector<vector<int>>& img, int x, int y, int size, double& grad_h, double& grad_v) {
    grad_h = 0.0;
    grad_v = 0.0;
    // 簡單計算：累加相鄰像素的差值絕對值
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (j + 1 < size) grad_h += abs(img[y + i][x + j] - img[y + i][x + j + 1]);
            if (i + 1 < size) grad_v += abs(img[y + i][x + j] - img[y + i + 1][x + j]);
        }
    }
    // 正規化避免 size 影響
    grad_h /= (size * size);
    grad_v /= (size * size);
}

// ============================================================
// RDO 計算函數 (真正的 Ground Truth 決策邏輯)
// ============================================================

// 計算區塊的平均值（作為預測值）
double compute_mean(const vector<vector<int>>& img, int x, int y, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            sum += img[y + i][x + j];
        }
    }
    return sum / (size * size);
}

// 計算 SSD (Sum of Squared Differences) - Distortion
double compute_SSD(const vector<vector<int>>& img, int x, int y, int size, double pred_value) {
    double ssd = 0.0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double diff = img[y + i][x + j] - pred_value;
            ssd += diff * diff;
        }
    }
    return ssd;
}

// 估算編碼所需的 bits (Rate)
double estimate_rate(double variance, int size) {
    double base_rate = log2(variance + 1.0) * size;
    double overhead = 2.0;
    return base_rate + overhead;
}

// RDO Result 結構
struct RDOResult {
    double cost;
    bool should_split;
};

// 前向宣告
double compute_split_cost(const vector<vector<int>>& img, int x, int y, int size);

// 計算不切分的 RD Cost
double compute_no_split_cost(const vector<vector<int>>& img, int x, int y, int size) {
    double pred = compute_mean(img, x, y, size);
    double distortion = compute_SSD(img, x, y, size, pred);
    double variance = compute_variance(img, x, y, size);
    double rate = estimate_rate(variance, size);
    return distortion + LAMBDA * rate;
}

// RDO 決策：比較切分 vs 不切分的 Cost
RDOResult rdo_decision(const vector<vector<int>>& img, int x, int y, int size) {
    double no_split_cost = compute_no_split_cost(img, x, y, size);
    
    if (size <= MIN_BLOCK_SIZE) {
        return {no_split_cost, false};
    }
    
    double split_cost = compute_split_cost(img, x, y, size);
    
    if (split_cost < no_split_cost) {
        return {split_cost, true};
    } else {
        return {no_split_cost, false};
    }
}

// 計算切分後四個子區塊的總 Cost
double compute_split_cost(const vector<vector<int>>& img, int x, int y, int size) {
    int half = size / 2;
    double total_cost = 0.0;
    
    total_cost += rdo_decision(img, x, y, half).cost;
    total_cost += rdo_decision(img, x + half, y, half).cost;
    total_cost += rdo_decision(img, x, y + half, half).cost;
    total_cost += rdo_decision(img, x + half, y + half, half).cost;
    
    total_cost += 1.0;  // 分割標誌的 bits
    
    return total_cost;
}

// ============================================================
// 遞迴收集訓練資料（用 RDO 決策作為 Ground Truth）
// ============================================================

void recursive_partition(const vector<vector<int>>& img, int x, int y, int size) {
    // 1. 計算特徵 (Features) - 這些是 ML 的輸入
    double var = compute_variance(img, x, y, size);
    double gh, gv;
    compute_gradients(img, x, y, size, gh, gv);

    // 2. 用 RDO 決定是否切分 (這是 Ground Truth！)
    RDOResult rdo = rdo_decision(img, x, y, size);
    bool should_split = rdo.should_split;

    // 3. 收集數據：特徵 + RDO 決策結果
    dataset.push_back({var, gh, gv, size, should_split ? 1 : 0});

    // 4. 根據 RDO 決策執行動作
    if (should_split) {
        int half = size / 2;
        recursive_partition(img, x, y, half);
        recursive_partition(img, x + half, y, half);
        recursive_partition(img, x, y + half, half);
        recursive_partition(img, x + half, y + half, half);
    }
}

// --- 匯出 CSV 給 Python 使用 ---
void export_csv(const string& filename) {
    ofstream file(filename);
    file << "variance,grad_h,grad_v,size,label\n"; // Header
    for (const auto& data : dataset) {
        file << data.variance << "," 
             << data.grad_h << "," 
             << data.grad_v << "," 
             << data.size << "," 
             << data.should_split << "\n";
    }
    file.close();
    cout << "Dataset exported to " << filename << " with " << dataset.size() << " samples.\n";
}

int main() {
    // 1. 準備圖片
    auto img = generate_dummy_image(IMG_SIZE);
    
    cout << "=== Ground Truth Generation (RDO-based) ===\n";
    cout << "Image size: " << IMG_SIZE << "x" << IMG_SIZE << "\n";
    cout << "Lambda (RDO): " << LAMBDA << "\n";
    cout << "Min block size: " << MIN_BLOCK_SIZE << "\n\n";

    // 2. 執行 RDO 分割 (Ground Truth Generation)
    recursive_partition(img, 0, 0, IMG_SIZE);

    // 3. 匯出訓練資料
    export_csv("block_data.csv");

    cout << "\nPhase 1 Complete. Ground Truth generated using RDO decisions.\n";
    return 0;
}