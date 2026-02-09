#!/usr/bin/env python3
"""
Run Benchmark Multiple Times and Generate Averaged Results

This script:
1. Executes main_hybrid N times
2. Parses the output and calculates averaged metrics
3. Saves the averaged pareto_data.csv
4. Generates visualization charts
"""

import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

NUM_RUNS = 3  # 測試次數

def run_benchmark():
    """執行一次 benchmark 並解析結果"""
    result = subprocess.run(
        ['./main_hybrid'],
        capture_output=True,
        text=True,
        cwd='/Users/hongyanmac/CS/Fast Block Partitioning'
    )
    return result.stdout + result.stderr

def parse_pareto_data(output):
    """解析 Pareto Frontier 數據"""
    data = []
    lines = output.split('\n')
    in_pareto = False
    
    for line in lines:
        if 'Threshold    Time(us)' in line:
            in_pareto = True
            continue
        if in_pareto and '----' in line:
            continue
        if in_pareto and '====' in line:
            break
        if in_pareto:
            # 解析格式: 0.50        8.17      5.56x      15.34%         0
            match = re.match(r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)%\s+(\d+)', line)
            if match:
                data.append({
                    'threshold': float(match.group(1)),
                    'time_us': float(match.group(2)),
                    'speedup': float(match.group(3)),
                    'accuracy': float(match.group(4)),
                    'rdo_ops': int(match.group(5))
                })
    
    return data

def parse_c_model_time(output):
    """解析 C Model baseline 時間"""
    # 格式: Time: 52.75 us (在 [C Model] 區塊內)
    match = re.search(r'\[C Model\].*?Time:\s+([\d.]+)\s*us', output, re.DOTALL)
    if match:
        return float(match.group(1))
    return None

def main():
    print(f"🚀 Running benchmark {NUM_RUNS} times...")
    print("=" * 60)
    
    all_runs = []
    c_model_times = []
    
    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...", end=" ", flush=True)
        output = run_benchmark()
        data = parse_pareto_data(output)
        c_time = parse_c_model_time(output)
        
        if data:
            all_runs.append(data)
            if c_time:
                c_model_times.append(c_time)
            print(f"✓ ({len(data)} data points, C Model: {c_time if c_time else 'N/A'} us)")
        else:
            print("✗ Parse failed!")
            print("  Output:", output[:500])
    
    if not all_runs:
        print("❌ No valid runs!")
        return
    
    # 計算平均值
    print("\n📊 Calculating averages...")
    
    averaged_data = []
    num_thresholds = len(all_runs[0])
    
    for idx in range(num_thresholds):
        threshold = all_runs[0][idx]['threshold']
        
        times = [run[idx]['time_us'] for run in all_runs]
        speedups = [run[idx]['speedup'] for run in all_runs]
        accuracies = [run[idx]['accuracy'] for run in all_runs]
        rdo_ops = all_runs[0][idx]['rdo_ops']  # RDO ops 是固定的
        
        avg_time = np.mean(times)
        avg_speedup = np.mean(speedups)
        std_speedup = np.std(speedups)
        avg_accuracy = accuracies[0]  # Accuracy 是固定的
        
        averaged_data.append({
            'threshold': threshold,
            'time_us': avg_time,
            'speedup': avg_speedup,
            'speedup_std': std_speedup,
            'accuracy': avg_accuracy,
            'rdo_ops': rdo_ops
        })
        
        print(f"  θ={threshold:.2f}: Speedup={avg_speedup:.2f}x ± {std_speedup:.2f}, Accuracy={avg_accuracy:.2f}%")
    
    avg_c_model_time = np.mean(c_model_times)
    print(f"\n  C Model baseline: {avg_c_model_time:.2f} us (avg)")
    
    # 儲存 CSV
    df = pd.DataFrame(averaged_data)
    df.to_csv('pareto_data.csv', index=False)
    print(f"\n✓ Saved: pareto_data.csv")
    
    # 生成圖表
    generate_charts(df, avg_c_model_time, NUM_RUNS)
    
    # 顯示結果表格
    print("\n" + "=" * 70)
    print(f"   AVERAGED RESULTS ({NUM_RUNS} runs)")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Time(μs)':>10} {'Speedup':>12} {'Accuracy':>10} {'RDO Ops':>10}")
    print("-" * 70)
    for row in averaged_data:
        print(f"{row['threshold']:>10.2f} {row['time_us']:>10.2f} {row['speedup']:>8.2f}x ± {row['speedup_std']:.2f} {row['accuracy']:>9.2f}% {row['rdo_ops']:>10}")
    print("=" * 70)

def generate_charts(df, c_model_time, num_runs):
    """生成 Pareto Frontier 圖表"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============================================
    # 圖 1: Pareto Frontier (Speedup vs Accuracy)
    # ============================================
    ax1 = axes[0]
    
    # 繪製曲線（帶誤差棒）
    ax1.errorbar(df['speedup'], df['accuracy'], 
                 xerr=df['speedup_std'],
                 fmt='o-', color='#2E86AB', 
                 linewidth=2, markersize=10, 
                 markerfacecolor='white', markeredgewidth=2,
                 capsize=4, capthick=1.5,
                 label='Hybrid Model')
    
    # 標註每個點的 threshold
    for i, row in df.iterrows():
        offset = (0.08, 2) if row['threshold'] != 0.6 else (-0.15, -5)
        ax1.annotate(f"θ={row['threshold']}", 
                     (row['speedup'], row['accuracy']),
                     textcoords="offset points", xytext=offset,
                     fontsize=9, color='#333')
    
    # 標記最佳平衡點 (θ=0.9)
    best_df = df[df['threshold'] == 0.90]
    if not best_df.empty:
        ax1.scatter([best_df.iloc[0]['speedup']], [best_df.iloc[0]['accuracy']], 
                    s=200, c='#E94F37', zorder=5, marker='*', 
                    label='Best Balance (θ=0.9)')
    
    ax1.set_xlabel('Speedup (×)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Pareto Frontier: Speedup vs Accuracy\n(Averaged over {num_runs} runs)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    
    # 動態設定 X 軸範圍
    max_speedup = df['speedup'].max()
    min_speedup = df['speedup'].min()
    ax1.set_xlim(min(0.5, min_speedup - 0.5), max_speedup + 1)
    ax1.set_ylim(0, 105)
    
    # 添加區域標註
    ax1.axhspan(80, 105, alpha=0.1, color='green')
    ax1.axvspan(2, max_speedup + 1, alpha=0.1, color='blue')
    ax1.text(max_speedup - 1, 25, 'Fast but\nInaccurate', fontsize=10, ha='center', color='#666')
    ax1.text(min(1.2, min_speedup + 0.3), 90, 'Accurate but\nSlow', fontsize=10, ha='center', color='#666')
    
    # 添加 baseline 參考線 (1.0x)
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(1.02, 5, 'Baseline\n(1.0x)', fontsize=8, color='gray')
    
    # ============================================
    # 圖 2: Threshold vs Metrics (雙軸圖)
    # ============================================
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Speedup 曲線（帶誤差棒）
    line1 = ax2.errorbar(df['threshold'], df['speedup'], 
                          yerr=df['speedup_std'],
                          fmt='s-', color='#2E86AB', 
                          linewidth=2, markersize=8, 
                          capsize=3, label='Speedup')
    ax2.set_xlabel('Confidence Threshold (θ)', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12, color='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    
    # baseline 參考線
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Accuracy 曲線
    line2, = ax2_twin.plot(df['threshold'], df['accuracy'], 'o-', color='#E94F37', 
                       linewidth=2, markersize=8, label='Accuracy')
    ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, color='#E94F37')
    ax2_twin.tick_params(axis='y', labelcolor='#E94F37')
    
    # 合併圖例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    ax2.set_title(f'Effect of Confidence Threshold\n(Error bars: ±1 std over {num_runs} runs)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0.45, 1.05)
    
    # 標記建議區間
    ax2.axvspan(0.8, 0.95, alpha=0.2, color='yellow')
    ax2.text(0.875, ax2.get_ylim()[0] + 0.5, 'Recommended\nZone', fontsize=9, ha='center', color='#666')
    
    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('pareto_frontier.pdf', bbox_inches='tight')
    
    print("\n✓ Charts saved:")
    print("  - pareto_frontier.png")
    print("  - pareto_frontier.pdf")

if __name__ == '__main__':
    main()
