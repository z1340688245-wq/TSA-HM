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
    r"\Data\Output\SingleDisease\Method\Road\L_TST.xes"
)

MODEL_DIR = (
    r"D:\pm_cp_adaptation-main\pm_cp_adaptation-main"
    r"\MyCleanModels\Road"
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


        event_count = len(log)

        trace_count = log[
            "case:concept:name"
        ].nunique()

        event_log = (
            pm4py.convert_to_event_log(log)
        )

    elif isinstance(log, EventLog):

        event_log = log

    else:

        try:

            event_log = (
                pm4py.convert_to_event_log(log)
            )

        except Exception as exc:

            raise TypeError(
            ) from exc

    if not isinstance(event_log, EventLog):

        event_log = EventLog(
            list(event_log)
        )

    return event_log


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


def build_evaluation_log(
        log,
        max_traces=MAX_EVAL_TRACES,
        seed=EVAL_SEED
):

    log = ensure_event_log(log)

    all_traces = list(log)

    trace_count = len(all_traces)

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

    print(
        "\n开始计算Alignment Fitness..."
    )

    fitness_result = (
        pm4py.fitness_alignments(
            eval_log,
            net,
            im,
            fm
        )
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
                f"{list(fitness_result.keys())}"
            )

    else:

        fitness = fitness_result

    precision_result = (
        pm4py.precision_alignments(
            eval_log,
            net,
            im,
            fm
        )
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
                f"{list(precision_result.keys())}"
            )

    else:

        precision = precision_result

    fitness = float(fitness)
    precision = float(precision)

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

    print("\n===== Road 测试 =====")

    # ---------------------------
    # 配置
    # ---------------------------

    cfg = load_config()

    # ---------------------------
    # 读取测试日志
    # ---------------------------

    print(
        "\n===== 读取测试日志 ====="
    )

    raw_test_log = pm4py.read_xes(
        TEST_XES
    )

    test_log = ensure_event_log(
        raw_test_log
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
            "===== Road =====\n"
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
