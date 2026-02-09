import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# =========================================================
# Phase 2: ML Model Training with Confidence Output
# =========================================================

# 1. 載入 Phase 1 產生的 Ground Truth 數據
# ---------------------------------------------------------
print("=" * 60)
print("Phase 2: Training Binary Model with Confidence")
print("=" * 60)

print("\n[1/4] Loading dataset...")
try:
    df = pd.read_csv("block_data.csv")
except FileNotFoundError:
    print("Error: 'block_data.csv' not found. Please run the Phase 1 C++ code first.")
    exit()

print(f"      Loaded {len(df)} samples")
print(f"      Split ratio: {df['label'].mean()*100:.1f}% split / {(1-df['label'].mean())*100:.1f}% no-split")

# 特徵與標籤
feature_names = ['variance', 'grad_h', 'grad_v']
X = df[feature_names]
y = df['label'] # 1 = Split, 0 = No Split

# 切分訓練/測試集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 訓練模型 (The "Binary Model")
# ---------------------------------------------------------
print("\n[2/4] Training Decision Tree Classifier...")
# 關鍵參數：max_depth=4 (平衡準確度與推論速度)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 驗證準確度
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"      Model Accuracy: {acc*100:.2f}%")

# 3. 分析 Confidence 分布
# ---------------------------------------------------------
print("\n[3/4] Analyzing Confidence Distribution...")
y_proba = clf.predict_proba(X_test)
confidences = np.max(y_proba, axis=1)
print(f"      Avg Confidence: {confidences.mean()*100:.1f}%")
print(f"      Min Confidence: {confidences.min()*100:.1f}%")
print(f"      Max Confidence: {confidences.max()*100:.1f}%")

# 4. 將 Sklearn Tree 轉譯為帶 Confidence 的 C++ 程式碼
# ---------------------------------------------------------
print("\n[4/4] Generating C++ Code with Confidence...")
print("-" * 60)

def tree_to_cpp_with_confidence(tree, feature_names):
    """
    將 Decision Tree 轉譯為 C++ 程式碼
    返回 struct { bool should_split; double confidence; }
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    lines = []
    
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # 決策節點 (if-else)
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append(f"{indent}if ({name} <= {threshold:.6f}) {{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            # 葉節點 -> 輸出預測結果與信心度
            value = tree_.value[node][0]
            class_idx = int(value.argmax())
            total_samples = value.sum()
            confidence = value[class_idx] / total_samples
            
            lines.append(f"{indent}// Samples: no_split={int(value[0])}, split={int(value[1])}")
            lines.append(f"{indent}return {{{'true' if class_idx == 1 else 'false'}, {confidence:.4f}}};")

    # 生成程式碼
    lines.append("// ============================================")
    lines.append("// Auto-generated from Decision Tree (max_depth=4)")
    lines.append("// ============================================")
    lines.append("")
    lines.append("struct MLPrediction {")
    lines.append("    bool should_split;")
    lines.append("    double confidence;")
    lines.append("};")
    lines.append("")
    lines.append("MLPrediction predict_split_ml(double variance, double grad_h, double grad_v) {")
    recurse(0, 1)
    lines.append("}")
    
    return "\n".join(lines)

cpp_code = tree_to_cpp_with_confidence(clf, feature_names)
print(cpp_code)
print("-" * 60)

# 5. 將程式碼寫入檔案
# ---------------------------------------------------------
with open("ml_model_generated.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write(cpp_code)

print(f"\n✓ C++ code saved to 'ml_model_generated.h'")
print("=" * 60)