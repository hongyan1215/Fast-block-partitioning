#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;

// --- 設定參數 ---
const int IMG_SIZE = 64;       // 圖片大小 (64x64)
const int MIN_BLOCK_SIZE = 4;  // 最小切分單位
const double SPLIT_THRESHOLD = 20.0; // 變異數閾值 (大於此值就切)

// --- 資料結構：用來存給 Python 訓練的資料 ---
struct TrainingData {
    double variance;
    double grad_h; // 水平梯度強度
    double grad_v; // 垂直梯度強度
    int size;
    int should_split; // Label: 1 = 切, 0 = 不切
};

vector<TrainingData> dataset;

// --- 輔助函數：生成一張測試用的灰階圖 ---
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

// --- 核心數學：計算區塊內的變異數 (Variance) ---
// 這是 "Heavy C Model" 模擬的部分，計算量較大
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
// 用於 Phase 2 的 Feature
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
    // 正規化一下，避免 size 影響太大
    grad_h /= (size * size);
    grad_v /= (size * size);
}

// --- 遞迴邏輯：Quadtree Partitioning ---
void recursive_partition(const vector<vector<int>>& img, int x, int y, int size) {
    // 1. 計算特徵 (Features)
    double var = compute_variance(img, x, y, size);
    double gh, gv;
    compute_gradients(img, x, y, size, gh, gv);

    // 2. 判斷是否需要切分 (Logic)
    bool should_split = (var > SPLIT_THRESHOLD) && (size > MIN_BLOCK_SIZE);

    // 3. 收集數據 (蒐集 Ground Truth)
    // 注意：我們只紀錄「當下這個 block」的決策，這就是 Python 要學的
    dataset.push_back({var, gh, gv, size, should_split ? 1 : 0});

    // 4. 執行動作
    if (should_split) {
        int half = size / 2;
        // 遞迴呼叫四個子區塊
        recursive_partition(img, x, y, half);             // Top-Left
        recursive_partition(img, x + half, y, half);      // Top-Right
        recursive_partition(img, x, y + half, half);      // Bottom-Left
        recursive_partition(img, x + half, y + half, half); // Bottom-Right
    } else {
        // Leaf Node: 這裡就是最終編碼的 block
        // 在真實 Encoder 這裡會進行 DCT/Quantization
        // std::cout << "Leaf Block at (" << x << "," << y << ") size: " << size << "\n";
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
    
    cout << "Start Ground Truth Generation (C Model)...\n";

    // 2. 執行暴力分割 (Ground Truth Generation)
    // 從整張圖開始切 (0, 0, 64)
    recursive_partition(img, 0, 0, IMG_SIZE);

    // 3. 匯出訓練資料
    export_csv("block_data.csv");

    cout << "Phase 1 Complete. Ready for Python training.\n";
    return 0;
}