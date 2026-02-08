# Fast Block Partitioning

使用機器學習加速影像/視訊編碼器中的區塊分割決策（Block Partitioning Decision）。

## 專案架構

```
Fast Block Partitioning/
├── ground_truth.cpp      # Phase 1: Ground Truth 資料生成
├── train_and_convert.py  # Phase 2: ML 模型訓練 & C++ 轉譯
├── main_hybrid.cpp       # Phase 3: Baseline vs Hybrid 效能比較
├── block_data.csv        # 訓練資料集 (自動生成)
└── README.md
```

## 工作流程

### Phase 1: Ground Truth Generation (C++)
執行 Quadtree Partitioning 演算法，收集區塊特徵與切分決策作為訓練資料。

```bash
g++ -o ground_truth ground_truth.cpp && ./ground_truth
```

**輸出：** `block_data.csv` (217 筆訓練樣本)

| 特徵 | 說明 |
|------|------|
| variance | 區塊像素變異數 |
| grad_h | 水平梯度強度 |
| grad_v | 垂直梯度強度 |
| size | 區塊大小 (4-64) |
| label | 是否切分 (0/1) |

### Phase 2: ML Model Training (Python)
使用 Decision Tree 訓練二元分類器，並自動轉譯為 C++ if-else 程式碼。

```bash
python train_and_convert.py
```

**模型設定：**
- Algorithm: Decision Tree Classifier
- Max Depth: 3 (確保 Branch Prediction 效率)
- Accuracy: **70.45%**

**生成的 C++ 決策邏輯：**
```cpp
bool predict_split_ml(double variance, double grad_h, double grad_v) {
    if (variance <= 5141.6001f) {
        if (grad_h <= 62.2188f) {
            if (grad_h <= 18.2188f) {
                return 0;  // Confidence 0.80
            } else {
                return 0;  // Confidence 1.00
            }
        } else {
            return 1;  // Confidence 1.00
        }
    } else {
        if (variance <= 5754.1450f) {
            if (variance <= 5577.6599f) {
                return 0;  // Confidence 0.61
            } else {
                return 1;  // Confidence 0.85
            }
        } else {
            if (variance <= 8911.4854f) {
                return 0;  // Confidence 0.93
            } else {
                return 1;  // Confidence 1.00
            }
        }
    }
}
```

### Phase 3: Integration & Benchmark (C++)
比較 Baseline (完整 RDO 計算) 與 Hybrid (ML 加速) 的效能差異。

```bash
g++ -O2 -o main_hybrid main_hybrid.cpp && ./main_hybrid
```

## 實驗結果

| 指標 | Baseline (Full RDO) | Hybrid (ML) |
|------|---------------------|-------------|
| 執行時間 | 2511.38 μs | 62.37 μs |
| 切分次數 | 54 | 13 |

### 關鍵指標

| 效能指標 | 數值 |
|----------|------|
| **加速比 (Speedup)** | **40.26x** |
| 結構匹配度 | ~24% |

## 技術特點

1. **Binary Model Design**: 將 ML 模型編譯為純 C++ if-else 結構，零運行時依賴
2. **Branch Prediction Friendly**: 限制決策樹深度，優化 CPU 分支預測
3. **Hybrid Architecture**: 結合 ML 快速推論與傳統演算法的可靠性

## 待改進

- [ ] 增加訓練資料多樣性（不同圖片類型）
- [ ] 調整 Decision Tree 深度以平衡速度與準確度
- [ ] 加入 Confidence-based Fallback 機制
- [ ] 實測真實視訊編碼器整合效果

## 環境需求

- C++ Compiler (支援 C++11)
- Python 3.x
- pandas, scikit-learn
