import os
import json
import pm4py
import warnings
import random
import pandas as pd

from pm4py.objects.log.obj import EventLog


warnings.filterwarnings("ignore")

EVAL_SEED = 99

random.seed(EVAL_SEED)


# =====================================================
# 路径
# =====================================================

TEST_XES = (
    r"D:\pm_cp_adaptation-main\pm_cp_adaptation-main"
    r"\Data\Output\SingleDisease\Method\BPIC2020\L_TST.xes"
)

MODEL_DIR = (
    r"D:\pm_cp_adaptation-main\pm_cp_adaptation-main"
    r"\MyCleanModels\BPIC_2020"
)

PNML_FILE = os.path.join(
    MODEL_DIR,
    "model.pnml"
)

CFG_FILE = os.path.join(
    MODEL_DIR,
    "train_config.json"
)



def ensure_event_log(log):


    if isinstance(log, pd.DataFrame):

        required_columns = {
            "case:concept:name",
            "concept:name"
        }

        missing_columns = (
            required_columns
            - set(log.columns)
        )

        if missing_columns:

            raise ValueError(
                "XES读取后的DataFrame缺少必要字段: "
                f"{sorted(missing_columns)}"
            )

        event_count = len(log)

        trace_count = log[
            "case:concept:name"
        ].nunique()

        event_log = (
            pm4py.convert_to_event_log(log)
        )

    elif isinstance(log, EventLog):

        event_log = log

        print(
            "\nread_xes返回类型: EventLog"
        )

    else:
        try:

            event_log = (
                pm4py.convert_to_event_log(log)
            )

        except Exception as exc:

            raise TypeError(
                f"当前类型: {type(log)}"
            ) from exc

    if not isinstance(event_log, EventLog):
        event_log = EventLog(
            list(event_log)
        )

    if len(event_log) == 0:

        raise ValueError(
        )

    print(
        f"{len(event_log)}"
    )

    return event_log

# =====================================================
# 读取配置
# =====================================================

def load_config():

    if not os.path.exists(CFG_FILE):
        raise FileNotFoundError(
            f"找不到配置文件:\n{CFG_FILE}"
        )

    with open(
        CFG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        cfg = json.load(f)

    return cfg

# =====================================================
# 活动过滤
# =====================================================

def filter_test_log(
        log,
        keep_activities
):

    df = pm4py.convert_to_dataframe(log)

    required_columns = {
        "case:concept:name",
        "concept:name"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "测试日志缺少必要字段: "
            f"{sorted(missing_columns)}"
        )

    before_trace = df[
        "case:concept:name"
    ].nunique()

    before_event = len(df)

    before_activities = set(
        df["concept:name"].unique()
    )

    keep_activities = set(
        keep_activities
    )

    removed_activities = sorted(
        before_activities
        - keep_activities
    )

    # 保存原始事件顺序
    df["_original_event_order"] = range(
        len(df)
    )

    # 仅保留训练阶段允许的活动
    df = df[
        df["concept:name"].isin(
            keep_activities
        )
    ].copy()

    sort_columns = [
        "case:concept:name"
    ]

    if "time:timestamp" in df.columns:
        sort_columns.append(
            "time:timestamp"
        )

    sort_columns.append(
        "_original_event_order"
    )

    df = df.sort_values(
        sort_columns,
        kind="stable"
    )

    df = df.drop(
        columns=["_original_event_order"]
    )

    after_trace = df[
        "case:concept:name"
    ].nunique()

    after_event = len(df)

    trace_retention = (
        after_trace
        / max(before_trace, 1)
    )

    event_retention = (
        after_event
        / max(before_event, 1)
    )

    print("\n===== 测试日志活动投影 =====")

    print(
        f"投影前Trace数 : {before_trace}"
    )

    print(
        f"投影后Trace数 : {after_trace}"
    )

    print(
        f"Trace保留率   : "
        f"{trace_retention:.4f}"
    )

    print(
        f"投影前Event数 : {before_event}"
    )

    print(
        f"投影后Event数 : {after_event}"
    )

    print(
        f"Event保留率   : "
        f"{event_retention:.4f}"
    )

    print(
        f"删除活动类型数: "
        f"{len(removed_activities)}"
    )

    if removed_activities:

        print("测试日志中被删除的活动:")

        print(removed_activities)

    return pm4py.convert_to_event_log(
        df
    )


# =====================================================
# 评估指标
# =====================================================

def build_evaluation_log(
        log,
        max_traces=MAX_EVAL_TRACES,
        seed=EVAL_SEED
):

    log = ensure_event_log(log)

    all_traces = list(log)

    trace_count = len(all_traces)

    if trace_count == 0:

        raise ValueError(
            "测试日志不包含任何Trace"
        )

    sample_size = min(
        max_traces,
        trace_count
    )

    if sample_size < trace_count:

        rng = random.Random(seed)

        sampled_traces = rng.sample(
            all_traces,
            sample_size
        )

        eval_log = EventLog(
            sampled_traces
        )

        print(
            "\n测试日志较大，使用固定随机样本:"
        )

        print(
            f"原始Trace数 : {trace_count}"
        )

        print(
            f"评估Trace数 : {sample_size}"
        )

        print(
            f"随机种子    : {seed}"
        )

    else:

        eval_log = EventLog(
            all_traces
        )

        print(
            "\n使用全部测试Trace进行评估:"
        )

        print(
            f"评估Trace数 : {trace_count}"
        )

    return eval_log

def evaluate_model(
        log,
        net,
        im,
        fm
):


    eval_log = build_evaluation_log(
        log=log,
        max_traces=MAX_EVAL_TRACES,
        seed=EVAL_SEED
    )

    # =====================================================
    # Alignment-based Fitness
    # =====================================================

    fitness_result = (
        pm4py.fitness_alignments(
            eval_log,
            net,
            im,
            fm
        )
    )

    print(
        f"Fitness原始结果: "
        f"{fitness_result}"
    )

    if isinstance(
        fitness_result,
        dict
    ):

        if (
            "average_trace_fitness"
            in fitness_result
        ):

            fitness = fitness_result[
                "average_trace_fitness"
            ]

        elif "log_fitness" in fitness_result:

            fitness = fitness_result[
                "log_fitness"
            ]

        elif (
            "averageFitness"
            in fitness_result
        ):

            fitness = fitness_result[
                "averageFitness"
            ]

        else:

            raise KeyError(
                "无法从Fitness结果中提取指标。"
                f"当前返回字段: "
                f"{list(fitness_result.keys())}"
            )

    else:

        fitness = fitness_result

    # =====================================================
    # Alignment-based Precision
    # =====================================================

    precision_result = (
        pm4py.precision_alignments(
            eval_log,
            net,
            im,
            fm
        )
    )

    print(
        f"Precision原始结果: "
        f"{precision_result}"
    )

    if isinstance(
        precision_result,
        dict
    ):

        if "precision" in precision_result:

            precision = precision_result[
                "precision"
            ]

        elif (
            "averagePrecision"
            in precision_result
        ):

            precision = precision_result[
                "averagePrecision"
            ]

        else:

            raise KeyError(
                "无法从Precision结果中提取指标。"
                f"当前返回字段: "
                f"{list(precision_result.keys())}"
            )

    else:

        precision = precision_result

    fitness = float(fitness)
    precision = float(precision)

    # 防止异常返回值
    fitness = max(
        0.0,
        min(1.0, fitness)
    )

    precision = max(
        0.0,
        min(1.0, precision)
    )

    mean_quality = (
        fitness + precision
    ) / 2.0

    if fitness + precision > 0:

        f1 = (
            2.0
            * fitness
            * precision
            / (
                fitness
                + precision
            )
        )

    else:

        f1 = 0.0

    return (
        fitness,
        precision,
        mean_quality,
        f1,
        len(eval_log)
    )


# =====================================================
# 主函数
# =====================================================

def test_model():

    print("\n===== BPIC2020 测试 =====")

    # ---------------------------
    # 配置
    # ---------------------------

    cfg = load_config()

    keep_activities = cfg[
        "keep_activities"
    ]

    print(
        f"\n训练保留活动数: "
        f"{len(keep_activities)}"
    )

    print(
        "\n训练保留活动:"
    )

    print(
        keep_activities
    )

    # ---------------------------
    # 读取测试日志
    # ---------------------------

    print(
        "\n===== 读取测试日志 ====="
    )

    raw_test_log = pm4py.read_xes(
        TEST_XES
    )

    # 关键修复：
    # 无论read_xes返回DataFrame还是EventLog，
    # 都统一转换成EventLog
    test_log = ensure_event_log(
        raw_test_log
    )

    if FILTER_TEST_ACTIVITIES:

        print(
            "\n根据训练活动集合投影测试日志..."
        )

        test_log = filter_test_log(
            test_log,
            keep_activities
        )

        # filter_test_log返回后再次确保类型正确
        test_log = ensure_event_log(
            test_log
        )

    else:

        print(
            "\n不删除测试活动，"
            "直接使用原始测试日志评估。"
        )

        print(
            f"原始测试Trace数: "
            f"{len(test_log)}"
        )

    # ---------------------------
    # 读取PetriNet
    # ---------------------------

    print(
        "\n读取PetriNet..."
    )

    net, im, fm = pm4py.read_pnml(
        PNML_FILE
    )

    # ---------------------------
    # 评估
    # ---------------------------

    (
        fitness,
        precision,
        mean_quality,
        f1,
        evaluated_traces
    ) = evaluate_model(
        test_log,
        net,
        im,
        fm
    )

    # ---------------------------
    # 输出
    # ---------------------------

    print(
        "\n===== 测试结果 ====="
    )

    print(
        f"Fitness    : {fitness:.4f}"
    )

    print(
        f"Precision  : {precision:.4f}"
    )

    print(
        f"Mean Quality: {mean_quality:.4f}"
    )

    print(
        f"Evaluated Traces: {evaluated_traces}"
    )

    print(
        f"F1 Score   : {f1:.4f}"
    )

    print("\n模型复杂度:")

    print(
        f"Places      : {len(net.places)}"
    )

    print(
        f"Transitions : {len(net.transitions)}"
    )

    print(
        f"Arcs        : {len(net.arcs)}"
    )

    # ---------------------------
    # 保存结果
    # ---------------------------

    result_file = os.path.join(
        MODEL_DIR,
        "test_result.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "===== BPIC2020 =====\n"
        )

        f.write(
            f"Fitness      : {fitness:.4f}\n"
        )

        f.write(
            f"Precision    : {precision:.4f}\n"
        )

        f.write(
            f"Mean Quality : {mean_quality:.4f}\n"
        )

        f.write(
            f"F1 Score     : {f1:.4f}\n"
        )

        f.write(
            f"Evaluated Traces: {evaluated_traces}\n"
        )

        f.write(
            f"\nPlaces      : {len(net.places)}\n"
        )

        f.write(
            f"Transitions : {len(net.transitions)}\n"
        )

        f.write(
            f"Arcs        : {len(net.arcs)}\n"
        )

    print(
        f"\n结果已保存:\n{result_file}"
    )

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    test_model()
