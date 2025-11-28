import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 스타일 설정 (깔끔한 그래프를 위해 seaborn 테마 사용)
sns.set_theme(style="whitegrid")

# 데이터 로드
df = pd.read_csv("research_codes/benchmark_results.csv")

# True/False 데이터 분리 (Row 0: True, Row 1: False 가정)
# 실제 데이터가 여러 행이라면 조건에 맞게 필터링 필요
row_true = df[df["enable_weight_sharding"] == True].iloc[0]
row_false = df[df["enable_weight_sharding"] == False].iloc[0]

small_batch_size = row_true["small_batch_size"]


# 데이터 추출 헬퍼 함수 (Add, Transpose, Matmul 순서)
def get_large(row):
    return [row["large_add_time"], row["large_transpose_time"], row["large_pure_matmul_time"]]


def get_mini_fwd(row):
    return [row["mini_per_fwd_add_time"], row["mini_per_fwd_transpose_time"], row["mini_per_fwd_matmul_time"]]


def get_mini_seq(row):
    return [row["mini_add_time"], row["mini_transpose_time"], row["mini_pure_matmul_time"]]


# 그래프를 그릴 데이터 준비
data = {
    "Large Batch (True)": get_large(row_true),
    "Large Batch (False)": get_large(row_false),
    "Mini Batch Fwd (True)": get_mini_fwd(row_true),
    "Mini Batch Fwd (False)": get_mini_fwd(row_false),
    "Mini Batch Seq (True)": get_mini_seq(row_true),
    "Mini Batch Seq (False)": get_mini_seq(row_false),
}

# 2행 3열의 서브플롯 생성
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 시나리오 정의 (키 접두어, 그래프 제목)
scenarios = [
    ("Large Batch", "Large Batch Forward Time"),
    ("Mini Batch Fwd", "Mini Batch (Per Forward Pass)"),
    ("Mini Batch Seq", "Mini Batch (Per Sequence)"),
]

components = ["Add", "Transpose", "Matmul"]
colors = sns.color_palette("pastel")[:3]  # 부드러운 파스텔 톤 색상

# 모든 데이터에서 최대값 계산 (y축 통일용)
max_val = 0
for key in data:
    total_time = sum(data[key])
    if total_time > max_val:
        max_val = total_time

# 여유 공간 추가 (10%)
y_limit = max_val * 1.1

for i, (key_prefix, title_prefix) in enumerate(scenarios):
    # 윗 행 (Row 0): Weight Sharding = True
    ax_true = axes[0, i]
    vals_true = data[f"{key_prefix} (True)"]
    bottom = 0
    for j, val in enumerate(vals_true):
        ax_true.bar(["Time"], [val], label=components[j], color=colors[j], bottom=bottom)
        bottom += val
    ax_true.set_title(f"{title_prefix}\n(Weight Sharding=True)")
    ax_true.set_ylabel("Time (ms)")
    ax_true.set_ylim(0, y_limit)

    # 아래 행 (Row 1): Weight Sharding = False
    ax_false = axes[1, i]
    vals_false = data[f"{key_prefix} (False)"]
    bottom = 0
    for j, val in enumerate(vals_false):
        ax_false.bar(["Time"], [val], label=components[j], color=colors[j], bottom=bottom)
        bottom += val
    ax_false.set_title(f"{title_prefix}\n(Weight Sharding=False)")
    ax_false.set_ylim(0, y_limit)

# 범례 추가 (첫 번째 그래프에만 표시)
axes[0, 0].legend(loc="upper right", title="Component")

plt.tight_layout()
# plt.show() # 혹은
plt.savefig("research_codes/result.png")
