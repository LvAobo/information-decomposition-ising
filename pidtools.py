# python3

"""
All variable / column names in the example correspond to the original string
positions inside each outcome, i.e. the first bit → 'X', the second →
'Y', and the third → 'Z'.
"""

from __future__ import annotations
import itertools
from typing import List
import pandas as pd
import numpy as np

DataFrame = pd.DataFrame

# ---------------------------------------------------------------------------
# 原函数：将 Distribution 转换为联合分布 DataFrame
# ---------------------------------------------------------------------------
def distribution_to_dataframe(dist, var_names: List[str]) -> DataFrame:
    rows = []
    for outcome, p in zip(dist.outcomes, dist.probabilities):
        if len(outcome) != len(var_names):
            raise ValueError("Length mismatch between outcome and var_names")
        row = list(outcome) + [p]
        rows.append(row)

    columns = var_names + ["Pr"]
    df = pd.DataFrame(rows, columns=columns)
    df["Pr"] = df["Pr"] / df["Pr"].sum()
    return df

# ---------------------------------------------------------------------------
# 新函数：提取边缘分布并根据阶数 order 将源变量子集视作新变量生成新的联合分布 DataFrame
# ---------------------------------------------------------------------------
def df_to_new_df(df: DataFrame, src: List[str], tgt: str, order: int) -> DataFrame:
    """
    提取 src 和 tgt 的边缘分布，然后将 src 中所有大小为 order 的子集视作新组合变量，拼接取值字符串，输出新的 DataFrame。

    参数:
        df    : 包含至少 src 列、tgt 列和 Pr 列的 DataFrame
        src   : 源变量名列表
        tgt   : 目标变量列名
        order : 组合子集大小，必须在 [1, len(src)-1] 之间

    返回:
        ndf   : 新的联合分布 DataFrame，列顺序为 ['Pr'] + 所有 '(Si,Sj,...)' + [tgt]

    抛错:
        如果 order < 1 或 >= len(src)，则抛出 ValueError
    """
    # 校验 order
    if order < 1 or order > len(src):
        raise ValueError(f"阶数 order({order}) 必须在 1 到 len(src)({len(src)}) 之间")
    # 1) 提取 src + tgt 的边缘分布（groupby 并求和）
    df_m = df.groupby(src + [tgt], sort=False)["Pr"].sum().reset_index()

    # 2) 枚举 src 中大小为 order 的子集
    combos = list(itertools.combinations(src, order))
    # 3) 将 df_m 中 src 列值转换为整数字符串
    for v in src:
        df_m[v] = df_m[v].astype(str).astype(str)
    # 4) 对每个子集生成新列，格式 '(S1,S2)'，值如 '0,1'
    for subset in combos:
        col_name = f"({','.join(subset)})"
        df_m[col_name] = df_m.apply(
            lambda row: ",".join(row[v] for v in subset),
            axis=1
        )
    # 5) 构建新的列顺序：Pr 在最前，组合列其次，tgt 最后
    new_cols = ["Pr"] + [f"({','.join(sub)})" for sub in combos] + [tgt]
    ndf = df_m[new_cols].copy()
    # 6) 打印示例用于调试
    # print("生成的 ndf 示例：")
    # print(ndf.head())
    return ndf

# ---------------------------------------------------------------------------
# 随机生成联合分布的测试函数
# ---------------------------------------------------------------------------
def generate_random_df(var_names: List[str]) -> DataFrame:
    """
    随机生成给定变量列表的二进制联合分布 DataFrame。
    每个变量取值为 0 或 1，概率随机并归一化。

    参数:
        var_names: 包含所有变量（源变量和目标变量）的列表。

    返回:
        rand_df  : 生成的联合分布 DataFrame，列为 var_names + ['Pr']。
    """
    # 生成所有可能的二进制组合
    outcomes = list(itertools.product([0, 1,2], repeat=len(var_names)))
    probs = np.random.rand(len(outcomes))
    probs /= probs.sum()

    rows = []
    for outcome, p in zip(outcomes, probs):
        rows.append(list(outcome) + [p])
    columns = var_names + ["Pr"]
    rand_df = pd.DataFrame(rows, columns=columns)
    return rand_df


# ---------------------------------------------------------------------------
# 原函数：创建条件独立分布 Q(x', y', … | target)
# ---------------------------------------------------------------------------
def build_conditionally_independent_df(df: DataFrame, target_vars: List[str]) -> DataFrame:
    """
    基于原始 DataFrame，构造一个新的联合分布 Q，
    使得在给定 target_vars 时，所有非目标变量条件独立。
    列名保持与输入一致，不添加撇号。

    步骤:
      1) 统计目标的边缘分布 P(target_vars)
      2) 统计每个非目标变量在目标条件下的条件分布 P(v | target)
      3) 枚举所有组合，计算 Q = P(target) * ∏ P(v | target)
      4) 输出的列顺序为 ['Pr'] + other_vars + target_vars
    """
    if "Pr" not in df.columns:
        raise KeyError("输入 DataFrame 必须包含 'Pr' 列")
    # 识别非目标变量
    other_vars = [c for c in df.columns if c not in target_vars + ["Pr"]]
    # 1) 计算目标的边缘分布 P(target_vars)
    p_target = df.groupby(target_vars, sort=False)["Pr"].sum()
    # 2) 计算每个非目标变量在目标条件下的条件概率 P(v | target)
    cond_probs: dict = {}
    for tgt_vals, group in df.groupby(target_vars, sort=False):
        key = tgt_vals if isinstance(tgt_vals, tuple) else (tgt_vals,)
        cond_probs[key] = {}
        total = group["Pr"].sum()
        for var in other_vars:
            cond_probs[key][var] = group.groupby(var, sort=False)["Pr"].sum() / total
    # 3) 枚举所有可能取值组合，重建联合分布 Q
    new_rows = []
    for tgt_vals, p_t in p_target.items():
        key = tgt_vals if isinstance(tgt_vals, tuple) else (tgt_vals,)
        values_list = [cond_probs[key][v].index.tolist() for v in other_vars]
        for combo in itertools.product(*values_list):
            prob = p_t
            for v, val in zip(other_vars, combo):
                prob *= cond_probs[key][v][val]
            new_rows.append([prob, *combo, *key])
    # 4) 输出列名不带撇号，按原名称排序
    cols = ["Pr"] + other_vars + target_vars
    new_df = pd.DataFrame(new_rows, columns=cols)
    # 归一化概率，保证总和为 1
    new_df["Pr"] = new_df["Pr"] / new_df["Pr"].sum()
    # print(new_df.head())
    return new_df


# ---------------------------------------------------------------------------
# 信息论基础函数：熵/多元互信息/条件熵/条件互信息（支持多变量）
# ---------------------------------------------------------------------------

def _to_df(data):
    """
    内部工具：如果传入的是 DataFrame 则直接返回，否则按 Excel 文件读取。
    """
    return data if isinstance(data, pd.DataFrame) else pd.read_excel(data)


def entropy(data, vars: List[str]) -> float:
    """
    计算随机变量集合 vars 的联合熵 H(vars)。

    参数:
        data: DataFrame 或文件路径，需包含 vars 列和 Pr 列。
        vars: 变量名列表，至少包含一个元素。

    返回:
        H = -∑ p(v) log2 p(v)，其中 p(v) 是 vars 组合取值的概率。
    """
    df = _to_df(data)
    probs = df.groupby(vars, sort=False)["Pr"].sum().values
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()


def mutual_information(data, vars: List[str]) -> float:
    """
    计算多变量交互信息（intersection information） I(X1;...;Xn)，
    采用包含-排除法则：
      I(vars) = ∑_{k=1}^n (-1)^(k-1) * Σ_{U⊆vars, |U|=k} H(U)

    参数:
        data: 包含 vars 列和 Pr 列的 DataFrame 或文件路径。
        vars: 参与计算的变量名列表，长度>=1。

    返回:
        交互信息值（float）。
    """
    if not isinstance(vars, list) or len(vars) < 1:
        raise ValueError("vars 必须是至少含一个元素的列表")
    result = 0.0
    n = len(vars)
    # 包含-排除法则
    for k in range(1, n+1):
        for subset in itertools.combinations(vars, k):
            H_sub = entropy(data, list(subset))
            result += ((-1)**(k-1)) * H_sub
    return result


def conditional_entropy(data, target: str, given: List[str]) -> float:
    """
    计算条件熵 H(target | given)。

    参数:
        data  : DataFrame 或文件路径，需包含 target 与 given 列及 Pr 列。
        target: 目标变量名。
        given : 条件变量名列表。

    返回:
        H(target | given) = H(target ∪ given) - H(given)
    """
    return entropy(data, [target] + given) - entropy(data, given)


def conditional_mutual_information(data, target: str, xs: List[str], given: List[str]) -> float:
    """
    计算条件互信息 I(xs; target | given)。

    参数:
        data  : DataFrame 或文件路径。
        target: 目标变量名或列表。
        xs    : 要测量与 target 条件互信息的变量列表。
        given : 条件变量列表。

    返回:
        I(xs; target | given) = H(target | given) - H(target | given ∪ xs)
    """
    return conditional_entropy(data, target, given) - conditional_entropy(data, target, given + xs)

# ---------------------------------------------------------------------------
# 两源 PID：输入 src 列表和 tgt，输出 pd.Series
# ---------------------------------------------------------------------------
def two_source_pid(df: DataFrame, src: List[str], tgt: str) -> pd.Series:
    if len(src) != 2:
        raise ValueError("src 列表长度必须为 2")
    src1, src2 = src
    order = len(src) - 1
    ndf = df_to_new_df(df, src, tgt, order)
    #此处有为了ising代码进行临时修改
    q_df = build_conditionally_independent_df(ndf, [tgt])
    col1 = f"({src1})"
    col2 = f"({src2})"
    redundancy = mutual_information(q_df, [col1, col2])
    h_q = conditional_entropy(q_df, tgt, [col1, col2])
    h_p = conditional_entropy(df, tgt, [src1, src2])
    synergy = h_q - h_p
    unique_src1 = conditional_mutual_information(q_df, tgt, [col1], [col2])
    unique_src2 = conditional_mutual_information(q_df, tgt, [col2], [col1])
    result = pd.Series({
        'redundancy':   redundancy,
        'synergy':      synergy,
        'unique_src1':  unique_src1,
        'unique_src2':  unique_src2,
    })
    result.name = (src1, src2, tgt)
    return result
# ---------------------------------------------------------------------------
# total_syn_effect 计算函数（已独立）
# ---------------------------------------------------------------------------
def total_syn_effect(df: DataFrame, src: List[str], tgt: str) -> float:
    """
    计算基于 order=1 分布的总协同增益：
      total_syn_effect = H_Q1(tgt | 所有组合列) - H_Q0(tgt | 所有src)
    其中:
      Q1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
      Q0 = build_conditionally_independent_df(df, [tgt])
    返回：浮点值
    """
    # print(df_to_new_df(df, src, tgt, order=1))
    q_df_1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
    H_new = conditional_entropy(q_df_1, tgt, [col for col in q_df_1.columns if col not in ['Pr', tgt]])
    H_old = conditional_entropy(df, tgt, src)
    # print(H_new)
    return H_new - H_old

# ---------------------------------------------------------------------------
# 多源协同效应及各阶协同计算函数
# ---------------------------------------------------------------------------
def multi_source_syn(df: DataFrame, src: List[str], tgt: str) -> pd.Series:
    """
    计算多源协同信息，返回含 n、total_syn_effect 和各阶协同效应的 pd.Series：
      - 'n': 源变量个数
      - 'total_syn_effect': 基于 order=1 的协同增益 H_Q - H_P
      - 'order_k_syn': k 从 1 到 n，对应每阶的协同效应

    对于每个 k:
      1) df_prev_raw = df_to_new_df(df, src, tgt, order=k-1) 或原始 df
      2) df_prev = build_conditionally_independent_df(df_prev_raw, [tgt])
      3) df_curr_raw = df_to_new_df(df, src, tgt, order=k) 或原始 df
      4) df_curr = build_conditionally_independent_df(df_curr_raw, [tgt])
      5) H1 = H(tgt | 所有组合列) on df_prev
      6) H2 = H(tgt | 所有组合列) on df_curr
      order_k_syn = H1 - H2

    total_syn_effect 计算:
      Q_df_1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
      H_new = H(tgt | 所有组合列) on Q_df_1
      Q_df_0 = build_conditionally_independent_df(df, [tgt])
      H_old = H(tgt | src) on Q_df_0
      total_syn_effect = H_new - H_old

    返回:
        pd.Series：索引 ['n','total_syn_effect','order_1_syn',...,'order_n_syn']，
                   name 属性为 (tuple(src), tgt)
    """
    n = len(src)
    results: dict = {'n': n}
    # 各阶协同效应
    for k in range(n, n+1):
        # H1 分布
        if k > 1:
            df_prev_raw = df_to_new_df(df, src, tgt, order=k-1)
            df_prev = build_conditionally_independent_df(df_prev_raw, [tgt])
        else:
            df_prev = build_conditionally_independent_df(df, [tgt])
        H1_vars = [col for col in df_prev.columns if col not in ['Pr', tgt]]
        # H2 分布
        if k < n:
            df_curr_raw = df_to_new_df(df, src, tgt, order=k)
            df_curr = build_conditionally_independent_df(df_curr_raw, [tgt])
        else:
            df_curr = df
        H2_vars = [col for col in df_curr.columns if col not in ['Pr', tgt]]
        # 计算条件熵差
        H1 = conditional_entropy(df_prev, tgt, H1_vars)
        H2 = conditional_entropy(df_curr, tgt, H2_vars)
        results[f'order_{k}_syn'] = H1 - H2
        print(results)
    series = pd.Series(results)
    series.name = (tuple(src), tgt)
    return series


def multi_source_un(df: pd.DataFrame, src: List[str], tgt: str) -> pd.Series:
    """
    对每个源变量计算关于目标的特有信息（unique information）。

    参数
    ----
    df : pd.DataFrame
        原始联合概率分布 DataFrame，包含列 src...、tgt 和 'Pr'。
    src : List[str]
        源变量名列表，例如 ['S1','S2','S3']。
    tgt : str
        目标变量名，例如 'T'。

    返回
    ----
    pd.Series
        索引为 "Un(源→目标|其它源组合)"，值为对应的特有信息量。
        Series.name = (tuple(src), tgt)
    """
    results: dict[str, float] = {}

    # 1) 先把所有变量的取值统一为整数字符串，方便拼接
    df_str = df.copy()
    for v in src + [tgt]:
        df_str[v] = df_str[v].astype(int).astype(str)

    # 2) 遍历每个源变量，计算其特有信息
    for s in src:
        # 2.1) 构造其他源变量列表
        others = [v for v in src if v != s]
        # 2.2) 合成其他源变量的列名，比如 '(S2,S3,S4)'
        oth_col = f"({','.join(others)})"

        # 2.3) 在 df_str 上生成合并后的中间表
        df_m = df_str.copy()
        # 拼接其它源变量的取值，生成新列
        df_m[oth_col] = df_m.apply(
            lambda row: ",".join(row[v] for v in others),
            axis=1
        )
        # 聚合成三变量联合分布：s、oth_col、tgt
        df_m2 = (
            df_m
            .groupby([s, oth_col, tgt], sort=False)["Pr"]
            .sum()
            .reset_index()
        )

        # 2.4) 强制条件独立化：在给定 tgt 时，让 s 与 oth_col 独立
        # print(df_m2.head())
        q_df = build_conditionally_independent_df(df_m2, [tgt])

        # 2.5) 计算条件互信息 I(tgt; s | oth_col)
        #    = H(tgt | oth_col) - H(tgt | [s, oth_col])
        unq_val = conditional_mutual_information(
            q_df,
            tgt,
            [s],
            [oth_col]
        )

        # 2.6) 存储结果，索引名中包含具体的“其它源组合”
        results[f"Un({s}→{tgt}|{oth_col})"] = unq_val

    # 3) 返回带名称的 Series，方便后续合并为 MultiIndex DataFrame
    ser = pd.Series(results)
    ser.name = (tuple(src), tgt)
    return ser

# ---------------------------------------------------------------------------
# multi-source redundancy 计算函数（n ≥ 2）
# ---------------------------------------------------------------------------
def multi_source_red(df: pd.DataFrame, src: List[str], tgt: str) -> float:
    """
    计算多源冗余信息（multi-source redundancy）。

    步骤（与 total_syn_effect 类似）:
      1) 先把原始分布 df 经 order=1 的转换，得到含 '(Si)' 列的新表：
           df1_raw = df_to_new_df(df, src, tgt, order=1)
      2) 在给定 tgt 的条件下强制这些 '(Si)' 条件独立：
           q_df_1 = build_conditionally_independent_df(df1_raw, [tgt])
      3) 对 q_df_1 中所有 '(Si)' 求交互信息：
           R = I( (S1); (S2); … ; (Sn) )

    返回:
        冗余信息（float）
    """
    # 1)  order = 1 → 生成单源拼接列 '(Si)'
    df1_raw = df_to_new_df(df, src, tgt, order=1)

    # 2)  强制条件独立化
    q_df_1 = build_conditionally_independent_df(df1_raw, [tgt])

    # 3)  取出所有 '(Si)' 列名，然后做多元交互信息
    combo_cols = [c for c in q_df_1.columns if c.startswith('(') and c.endswith(')')]
    redundancy = mutual_information(q_df_1, combo_cols)

    return redundancy



if __name__ == "__main__":

    var_names = ["S1", "S2", "S3","S4",  "T"]
    src_vars = ["S1", "S2", "S3","S4"]
    tgt_var = "T"

    neg_cases: List[pd.DataFrame] = []  # 保存出现负冗余的分布
    for i in range(10000):
        df_rand = generate_random_df(var_names)
        R = multi_source_red(df_rand, src_vars, tgt_var)

        # 如果冗余为负就输出并存档
        if R < -0.0001:
            print(f"\n### Case {i:04d}  -- redundancy = {R:.6f}")
            print(df_rand.to_string(index=False))
            neg_cases.append(df_rand)

    print(f"\n共生成 1000 个随机分布，其中冗余为负的有 {len(neg_cases)} 个。")




