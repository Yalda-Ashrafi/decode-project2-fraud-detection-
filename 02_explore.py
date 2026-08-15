# 02_explore.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:/Decode_Intern/fraud-detection/creditcard.csv")


# ---- Chart 1: class imbalance ----
counts = df["Class"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: raw counts on a log scale, otherwise the fraud bar is invisible
axes[0].bar(["Legitimate (0)", "Fraud (1)"], counts.values,
            color=["#2a9d8f", "#e76f51"])
axes[0].set_yscale("log")
axes[0].set_ylabel("Number of transactions (log scale)")
axes[0].set_title("Class distribution")
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

# Right: percentage
pct = 100 * counts / counts.sum()
axes[1].pie(pct, labels=[f"Legitimate\n{pct[0]:.2f}%", f"Fraud\n{pct[1]:.2f}%"],
            colors=["#2a9d8f", "#e76f51"], startangle=90,
            explode=(0, 0.3), autopct=None)
axes[1].set_title("Proportion of each class")

plt.tight_layout()
plt.savefig("class_imbalance.png", dpi=150)
plt.show()
input("Press Enter to continue...")


# ---- Chart 2: does Amount differ between classes? ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cls, ax, title in zip([0, 1], axes, ["Legitimate", "Fraud"]):
    subset = df[df["Class"] == cls]["Amount"]
    ax.hist(subset, bins=50, range=(0, 500), color="#264653")
    ax.set_title(f"{title} — transaction amounts")
    ax.set_xlabel("Amount")
plt.tight_layout()
plt.show()
input("Press Enter to continue...")


print(df.groupby("Class")["Amount"].describe())

# ---- Chart 3: which features separate the classes best? ----
corr = df.corr()["Class"].drop("Class").sort_values()
plt.figure(figsize=(10, 6))
corr.plot(kind="barh", color=["#e76f51" if v < 0 else "#2a9d8f" for v in corr])
plt.title("Correlation of each feature with Class")
plt.tight_layout()
plt.show()
input("Press Enter to continue...")

