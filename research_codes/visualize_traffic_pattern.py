import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_mesh_grid(ax, rows, cols):
    """배경이 되는 NoC Core Mesh Grid를 그립니다."""
    for r in range(rows):
        for c in range(cols):
            rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor="#dee2e6", facecolor="#f8f9fa", zorder=1)
            ax.add_patch(rect)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
rows, cols = 6, 8  # 간략화된 칩 코어 그리드 (실제 120코어의 축소판)

# ==========================================
# 1. Sharded Layout (N-to-1 병목)
# ==========================================
ax1 = axes[0]
draw_mesh_grid(ax1, rows, cols)
ax1.set_title("Sharded Layout: N-on-1 Bottleneck", fontsize=16, fontweight="bold", pad=20)

target_r, target_c = 2, 6  # 거대 텐서를 들고 있는 타겟 코어

# 타겟 코어 하이라이트 및 거대 텐서 말풍선
target_rect = patches.Rectangle(
    (target_c, target_r), 1, 1, linewidth=2, edgecolor="#c92a2a", facecolor="#ffe3e3", zorder=2
)
ax1.add_patch(target_rect)
ax1.text(target_c + 0.5, target_r + 0.5, "Target\nCore", ha="center", va="center", color="#c92a2a", fontweight="bold")

# 텐서 형태 (말풍선/블록)
tensor_box = patches.FancyBboxPatch(
    (target_c - 1.5, target_r + 1.2), 3, 1.2, boxstyle="round,pad=0.1", ec="#c92a2a", fc="white", lw=1.5, zorder=3
)
ax1.add_patch(tensor_box)
ax1.text(
    target_c,
    target_r + 1.8,
    "Contiguous Tensor\n[1, Seq_Len, Head_Dim]",
    ha="center",
    va="center",
    fontsize=10,
    color="#c92a2a",
    fontweight="bold",
)

# 다른 코어들에서 타겟으로 쏠리는 화살표 (Inward)
for r in range(1, 5):
    for c in range(1, 4):
        # 연산 코어 하이라이트
        ax1.add_patch(patches.Rectangle((c, r), 1, 1, edgecolor="#868e96", facecolor="#e9ecef", zorder=2))
        # 트래픽 화살표 (집중)
        arrow = patches.FancyArrowPatch(
            (c + 0.5, r + 0.5),
            (target_c + 0.2, target_r + 0.5),
            connectionstyle="arc3,rad=0.1",
            color="#fa5252",
            alpha=0.7,
            arrowstyle="->",
            mutation_scale=15,
            lw=2,
            zorder=3,
        )
        ax1.add_patch(arrow)

# ==========================================
# 2. Interleaved Layout (1-to-N 분산)
# ==========================================
ax2 = axes[1]
draw_mesh_grid(ax2, rows, cols)
ax2.set_title("Interleaved Layout: 1-to-N Scatter-Gather", fontsize=16, fontweight="bold", pad=20)

compute_r, compute_c = 2, 2  # 연산을 수행하며 요청을 보내는 클라이언트 코어

# 클라이언트 코어 하이라이트
compute_rect = patches.Rectangle(
    (compute_c, compute_r), 1, 1, linewidth=2, edgecolor="#1864ab", facecolor="#e7f5ff", zorder=2
)
ax2.add_patch(compute_rect)
ax2.text(
    compute_c + 0.5, compute_r + 0.5, "Compute\nCore", ha="center", va="center", color="#1864ab", fontweight="bold"
)

# 칩 전역에 흩어진 타일들과 텐서 형태 설명
tile_coords = [(1, 1), (1, 5), (4, 1), (4, 6), (5, 3), (0, 4), (3, 7)]
for i, (r, c) in enumerate(tile_coords):
    # 타일 코어 하이라이트
    ax2.add_patch(patches.Rectangle((c, r), 1, 1, edgecolor="#2b8a3e", facecolor="#ebfbee", zorder=2))

    # 텐서 형태 (작은 타일)
    ax2.text(c + 0.5, r + 0.8, f"Tile {i}", ha="center", va="center", fontsize=8, color="#2b8a3e", fontweight="bold")
    ax2.text(c + 0.5, r + 0.2, "[32, 32]", ha="center", va="center", fontsize=7, color="#2b8a3e")

    # 연산 코어에서 전체 칩으로 뻗어 나가는 화살표 (Outward)
    arrow = patches.FancyArrowPatch(
        (compute_c + 0.8, compute_r + 0.5),
        (c + 0.5, r + 0.5),
        connectionstyle="arc3,rad=-0.1",
        color="#339af0",
        alpha=0.8,
        arrowstyle="->",
        mutation_scale=15,
        lw=1.5,
        zorder=3,
    )
    ax2.add_patch(arrow)

# 전체 레이아웃 조정 및 출력
plt.tight_layout()
plt.show()
plt.savefig("layout_comparison.png", dpi=300, bbox_inches="tight")  # 슬라이드 삽입용 고화질 저장
