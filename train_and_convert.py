import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 載入 Phase 1 產生的 Ground Truth 數據
# ---------------------------------------------------------
print("Loading dataset...")
try:
    df = pd.read_csv("block_data.csv")
except FileNotFoundError:
    print("Error: 'block_data.csv' not found. Please run the Phase 1 C++ code first.")
    exit()

# 特徵與標籤
feature_names = ['variance', 'grad_h', 'grad_v']
X = df[feature_names]
y = df['label'] # 1 = Split, 0 = No Split

# 切分訓練/測試集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 訓練模型 (The "Binary Model")
# ---------------------------------------------------------
# 關鍵參數：max_depth=3
# 這是為了保證生成的 C++ if-else 巢狀結構不會太深，保持極致的 Branch Prediction 效率。
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 驗證準確度
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Trained. Accuracy: {acc*100:.2f}%")
print("-" * 30)

# 3. 魔術時刻：將 Sklearn Tree 轉譯為 C++ 程式碼
# ---------------------------------------------------------
def tree_to_cpp(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    # 遞迴生成函數
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # 這是決策節點 (if-else)
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if ({name} <= {threshold:.4f}f) {{")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}}}")
        else:
            # 這是葉節點 (Leaf Node) -> 輸出預測結果
            # tree_.value[node] 是一個陣列，例如 [[10, 50]] 代表 [不切, 切] 的樣本數
            # 我們取 argmax 決定最終類別
            value = tree_.value[node][0]
            class_idx = int(value.argmax()) 
            
            # 這裡可以加入 "Confidence" 邏輯 (選擇性)
            # 例如：如果兩個類別數量差不多，代表信心不足
            total_samples = value.sum()
            confidence = value[class_idx] / total_samples
            
            print(f"{indent}// Leaf: Class {class_idx}, Confidence {confidence:.2f}")
            print(f"{indent}return {class_idx};")

    print("// --- Generated C++ Decision Tree Logic ---")
    print("bool predict_split(double variance, double grad_h, double grad_v) {")
    recurse(0, 1)
    print("}")
    print("// -------------------------------------------")

# 執行轉譯
tree_to_cpp(clf, feature_names)