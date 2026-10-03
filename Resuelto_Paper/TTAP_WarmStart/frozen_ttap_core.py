"""Shared TTAP definitions from 20260824_Experimento_B at the pinned commit.
Executed inside the selected generator namespace by ttap_warm_start.py.
This file does not execute an experiment on import.
"""

def helo_to_int(h):
    if isinstance(h, str):
        h = h.strip()
        return int(h[1:]) if h.startswith("H") else int(h)
    return int(h)

def euclidean_distance(scenario, node_a, node_b):
    xa, ya = scenario["nodes"][node_a]
    xb, yb = scenario["nodes"][node_b]
    return math.hypot(xa - xb, ya - yb)

def travel_time_minutes(scenario, h, node_a, node_b):
    if node_a == node_b:
        return 0.0
    distance = euclidean_distance(scenario, node_a, node_b)
    return 60.0 * distance / scenario["helicopters"][h]["speed"]

def logistic_component(alpha, t, t_eff):
    return 1.0 / (1.0 + math.exp(alpha * (t - t_eff)))

def satisfaction(scenario, t, task):
    """Anchored function: S(t_opt)=1, S(t_eff)=0.5, S(t_ineff)=0."""
    alpha = float(scenario["params"]["ALPHA"])
    t_opt = float(task["t_opt"])
    t_eff = float(task["t_eff"])
    t_ineff = float(task["t_ineff"])
    if not (0 <= t_opt < t_eff < t_ineff):
        raise ValueError("Temporal thresholds must satisfy 0 <= t_opt < t_eff < t_ineff.")
    if alpha <= 0:
        raise ValueError("ALPHA must be positive.")
    if t <= t_opt:
        return 1.0
    if t >= t_ineff:
        return 0.0

    value_t = logistic_component(alpha, t, t_eff)
    if t <= t_eff:
        value_opt = logistic_component(alpha, t_opt, t_eff)
        value = 0.5 + 0.5 * (value_t - 0.5) / (value_opt - 0.5)
    else:
        value_ineff = logistic_component(alpha, t_ineff, t_eff)
        value = 0.5 * (value_t - value_ineff) / (0.5 - value_ineff)
    return float(np.clip(value, 0.0, 1.0))

def task_benefit(scenario, task_id, finish_time):
    task = scenario["tasks"][task_id]
    return (
        task["potential_value"]
        * satisfaction(scenario, finish_time, task)
        / scenario["total_potential_value"]
    )

def can_execute_task_by_resources(scenario, h, task_id, load, med, pers):
    helo = scenario["helicopters"][h]
    task = scenario["tasks"][task_id]
    return (
        (not task["visual"] or helo["visual"])
        and load >= task["load"]
        and med >= task["med"]
        and pers >= task["pers"]
    )

def ceil_steps(minutes, time_unit):
    return int(math.ceil((minutes - 1e-9) / time_unit))

def time_from_step(k, time_unit):
    return k * time_unit

def discretized_finish(start_time, duration, time_unit):
    k_now = int(round(start_time / time_unit))
    return time_from_step(k_now + ceil_steps(duration, time_unit), time_unit)

def make_state(node, k, load, med, pers):
    return (node, int(k), int(load), int(med), int(pers))

def start_state(scenario, h):
    helo = scenario["helicopters"][h]
    return make_state("Base", 0, helo["load"], helo["med"], helo["pers"])

def sink_state():
    return ("SINK", -1, 0, 0, 0)

def validate_temporal_satisfaction(scenario):
    for task in scenario["tasks"].values():
        assert math.isclose(satisfaction(scenario, task["t_opt"], task), 1.0, abs_tol=1e-12)
        assert math.isclose(satisfaction(scenario, task["t_eff"], task), 0.5, abs_tol=1e-12)
        assert math.isclose(satisfaction(scenario, task["t_ineff"], task), 0.0, abs_tol=1e-12)
    return True

GREEDY_RULES = {
    "benefit": "Greedy-Benefit",
    "density_operational": "Greedy-Density-Operational",
}

def greedy_score(rule, benefit, operational_duration):
    if rule == "benefit":
        return benefit

    if rule == "density_operational":
        return benefit / operational_duration

    raise ValueError(f"Unknown greedy rule: {rule}")

def solve_greedy_ttap(scenario, rule="benefit", verbose=False):
    if rule not in GREEDY_RULES:
        raise ValueError(f"rule must be one of {sorted(GREEDY_RULES)}")
    method_name = GREEDY_RULES[rule]
    t0 = time.perf_counter()

    params = scenario["params"]
    tasks = scenario["tasks"]
    task_to_node = scenario["task_to_node"]
    helicopters = scenario["helicopters"]
    service_time = params["SERVICE_TIME"]
    recovery_time = params["RECOVERY_TIME"]
    tmax = params["TMAX"]
    time_unit = params["TIME_UNIT"]

    helo_state = {
        h: {
            "node": "Base",
            "available_time": 0.0,
            "load": helo["load"],
            "med": helo["med"],
            "pers": helo["pers"],
            "blocked": False,
        }
        for h, helo in helicopters.items()
    }
    pending = set(tasks)
    rows = []

    while pending:
        active_helos = [h for h in helicopters if not helo_state[h]["blocked"]]
        if not active_helos:
            break
        h = min(
            active_helos,
            key=lambda name: (helo_state[name]["available_time"], name),
        )
        state = helo_state[h]
        if state["available_time"] >= tmax:
            state["blocked"] = True
            continue

        current_node = state["node"]
        current_time = state["available_time"]
        candidates = []

        for task_id in sorted(pending):
            if not can_execute_task_by_resources(
                scenario,
                h,
                task_id,
                state["load"],
                state["med"],
                state["pers"],
            ):
                continue

            task_node = task_to_node[task_id]
            outbound_flight = travel_time_minutes(scenario, h, current_node, task_node)
            operational_duration = outbound_flight + service_time
            finish_time = discretized_finish(current_time, operational_duration, time_unit)
            if finish_time > tmax:
                continue

            return_flight = travel_time_minutes(scenario, h, task_node, "Base")
            return_duration = return_flight + recovery_time
            mission_finish_time = discretized_finish(finish_time, return_duration, time_unit)
            if mission_finish_time > tmax:
                continue

            benefit = task_benefit(scenario, task_id, finish_time)
            if benefit <= 0:
                continue

            cycle_duration = operational_duration + return_duration
            score = greedy_score(rule, benefit, operational_duration)
            candidates.append({
                "task_id": task_id,
                "task_node": task_node,
                "outbound_flight": outbound_flight,
                "return_flight": return_flight,
                "operational_duration": operational_duration,
                "cycle_duration": cycle_duration,
                "finish_time": finish_time,
                "mission_finish_time": mission_finish_time,
                "benefit": benefit,
                "score": score,
            })

        if candidates:
            best = min(
                candidates,
                key=lambda item: (
                    -item["score"],
                    -item["benefit"],
                    item["finish_time"],
                    item["task_id"],
                ),
            )
            task_id = best["task_id"]
            task = tasks[task_id]
            start_node = current_node
            start_time = current_time

            state["node"] = best["task_node"]
            state["available_time"] = best["finish_time"]
            state["load"] -= task["load"]
            state["med"] -= task["med"]
            state["pers"] -= task["pers"]
            pending.remove(task_id)

            rows.append({
                "Scenario": scenario["scenario_id"],
                "Method": method_name,
                "Helo ID": helo_to_int(h),
                "Task ID": task_id,
                "Task Class": task["priority_class"],
                "Start Node": start_node,
                "Node": best["task_node"],
                "Start Time": start_time,
                "Flight Time": best["outbound_flight"],
                "Service Time": service_time,
                "Total Task Time": best["operational_duration"],
                "Cycle Time Proxy": best["cycle_duration"],
                "Completion Time": best["finish_time"],
                "Benefit": best["benefit"],
                "Selection Score": best["score"],
                "Raw Impact": task["raw_impact"],
                "Normalized Impact": task["impact"],
                "Remaining Load": state["load"],
                "Remaining Medical": state["med"],
                "Remaining Personnel": state["pers"],
                "Status": "Task Executed",
            })
            if verbose:
                print(f"{method_name}: {h} -> T{task_id} at {best['finish_time']:.1f}")
            continue

        # No feasible positive-value task: recover if away from Base.
        if current_node != "Base":
            flight = travel_time_minutes(scenario, h, current_node, "Base")
            duration = flight + recovery_time
            finish_time = discretized_finish(current_time, duration, time_unit)
            if finish_time > tmax:
                state["blocked"] = True
                continue
            helo = helicopters[h]
            state.update({
                "node": "Base",
                "available_time": finish_time,
                "load": helo["load"],
                "med": helo["med"],
                "pers": helo["pers"],
            })
            rows.append({
                "Scenario": scenario["scenario_id"],
                "Method": method_name,
                "Helo ID": helo_to_int(h),
                "Task ID": "Idle",
                "Task Class": "return",
                "Start Node": current_node,
                "Node": "Base",
                "Start Time": current_time,
                "Flight Time": flight,
                "Service Time": recovery_time,
                "Total Task Time": duration,
                "Cycle Time Proxy": duration,
                "Completion Time": finish_time,
                "Benefit": 0.0,
                "Selection Score": 0.0,
                "Raw Impact": 0.0,
                "Normalized Impact": 0.0,
                "Remaining Load": state["load"],
                "Remaining Medical": state["med"],
                "Remaining Personnel": state["pers"],
                "Status": "Return/Recovery",
            })
        else:
            state["blocked"] = True

    # Final mandatory recovery for every used helicopter still away from Base.
    for h, state in helo_state.items():
        if state["node"] == "Base":
            continue
        current_node = state["node"]
        current_time = state["available_time"]
        flight = travel_time_minutes(scenario, h, current_node, "Base")
        duration = flight + recovery_time
        finish_time = discretized_finish(current_time, duration, time_unit)
        if finish_time > tmax:
            raise RuntimeError(f"{method_name}: {h} cannot complete mandatory recovery.")
        helo = helicopters[h]
        state.update({
            "node": "Base",
            "available_time": finish_time,
            "load": helo["load"],
            "med": helo["med"],
            "pers": helo["pers"],
        })
        rows.append({
            "Scenario": scenario["scenario_id"],
            "Method": method_name,
            "Helo ID": helo_to_int(h),
            "Task ID": "Idle",
            "Task Class": "return",
            "Start Node": current_node,
            "Node": "Base",
            "Start Time": current_time,
            "Flight Time": flight,
            "Service Time": recovery_time,
            "Total Task Time": duration,
            "Cycle Time Proxy": duration,
            "Completion Time": finish_time,
            "Benefit": 0.0,
            "Selection Score": 0.0,
            "Raw Impact": 0.0,
            "Normalized Impact": 0.0,
            "Remaining Load": state["load"],
            "Remaining Medical": state["med"],
            "Remaining Personnel": state["pers"],
            "Status": "Return/Recovery",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Completion Time", "Helo ID"]).reset_index(drop=True)
    summary = {
        "Scenario": scenario["scenario_id"],
        "Method": method_name,
        "Runtime sec": time.perf_counter() - t0,
        "Solver Time sec": np.nan,
        "Graph Time sec": np.nan,
        "Matrix Time sec": np.nan,
        "End-to-end sec": time.perf_counter() - t0,
        "Optimal Certified": False,
        "MIP Gap": np.nan,
        "Primal Bound": np.nan,
        "Dual Bound": np.nan,
        "Variables": np.nan,
        "Constraints": np.nan,
        "Nonzeros": np.nan,
        "Solver Message": "Constructive heuristic",
        "Has Incumbent": True,
    }
    return df, summary

def build_state_graph_for_helicopter(scenario, h):
    params = scenario["params"]
    tasks = scenario["tasks"]
    task_to_node = scenario["task_to_node"]
    helicopters = scenario["helicopters"]
    service_time = params["SERVICE_TIME"]
    recovery_time = params["RECOVERY_TIME"]
    tmax = params["TMAX"]
    time_unit = params["TIME_UNIT"]
    tmax_steps = int(round(tmax / time_unit))

    source = start_state(scenario, h)
    sink = sink_state()
    states = {source}
    queue = deque([source])
    arcs = []

    while queue:
        current_state = queue.popleft()
        node, k, load, med, pers = current_state
        current_time = time_from_step(k, time_unit)

        for task_id in sorted(tasks):
            if not can_execute_task_by_resources(scenario, h, task_id, load, med, pers):
                continue
            task_node = task_to_node[task_id]
            flight = travel_time_minutes(scenario, h, node, task_node)
            duration_steps = ceil_steps(flight + service_time, time_unit)
            k_new = k + duration_steps
            if k_new > tmax_steps:
                continue
            finish_time = time_from_step(k_new, time_unit)
            task = tasks[task_id]
            new_state = make_state(
                task_node,
                k_new,
                load - task["load"],
                med - task["med"],
                pers - task["pers"],
            )
            benefit = task_benefit(scenario, task_id, finish_time)
            if benefit <= 0:
                continue
            arcs.append({
                "h": h,
                "from": current_state,
                "to": new_state,
                "kind": "task",
                "task": task_id,
                "start_time": current_time,
                "finish_time": finish_time,
                "flight_time": flight,
                "service_time": service_time,
                "benefit": benefit,
            })
            if new_state not in states:
                states.add(new_state)
                queue.append(new_state)

        if node != "Base":
            flight = travel_time_minutes(scenario, h, node, "Base")
            duration_steps = ceil_steps(flight + recovery_time, time_unit)
            k_new = k + duration_steps
            if k_new <= tmax_steps:
                helo = helicopters[h]
                new_state = make_state(
                    "Base", k_new, helo["load"], helo["med"], helo["pers"]
                )
                arcs.append({
                    "h": h,
                    "from": current_state,
                    "to": new_state,
                    "kind": "return",
                    "task": None,
                    "start_time": current_time,
                    "finish_time": time_from_step(k_new, time_unit),
                    "flight_time": flight,
                    "service_time": recovery_time,
                    "benefit": 0.0,
                })
                if new_state not in states:
                    states.add(new_state)
                    queue.append(new_state)

    states.add(sink)
    full = helicopters[h]
    for state in list(states):
        if state == sink:
            continue
        node, k, load, med, pers = state
        recovered = (
            node == "Base"
            and load == full["load"]
            and med == full["med"]
            and pers == full["pers"]
        )
        if recovered:
            arcs.append({
                "h": h,
                "from": state,
                "to": sink,
                "kind": "end",
                "task": None,
                "start_time": time_from_step(k, time_unit),
                "finish_time": time_from_step(k, time_unit),
                "flight_time": 0.0,
                "service_time": 0.0,
                "benefit": 0.0,
            })
    return list(states), arcs

def build_full_exact_graph(scenario, verbose=False):
    states_by_h = {}
    arcs_by_h = {}
    all_arcs = []
    for h in scenario["helicopters"]:
        states, arcs = build_state_graph_for_helicopter(scenario, h)
        states_by_h[h] = states
        arcs_by_h[h] = arcs
        all_arcs.extend(arcs)
        if verbose:
            print(f"{h}: {len(states):,} states | {len(arcs):,} arcs")
    return states_by_h, arcs_by_h, all_arcs

def build_exact_milp_matrices(scenario, states_by_h, arcs_by_h, all_arcs):
    helicopters = scenario["helicopters"]
    tasks = scenario["tasks"]
    n_vars = len(all_arcs)
    c = np.array([-arc["benefit"] for arc in all_arcs], dtype=float)
    integrality = np.ones(n_vars, dtype=int)
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))

    row_idx, col_idx, values = [], [], []
    rhs_lb, rhs_ub = [], []
    current_row = 0

    offsets = {}
    offset = 0
    for h in helicopters:
        offsets[h] = offset
        offset += len(arcs_by_h[h])

    for h in helicopters:
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for local_index, arc in enumerate(arcs_by_h[h]):
            j = offsets[h] + local_index
            outgoing[arc["from"]].append(j)
            incoming[arc["to"]].append(j)

        source = start_state(scenario, h)
        sink = sink_state()
        for state in states_by_h[h]:
            for j in outgoing.get(state, []):
                row_idx.append(current_row); col_idx.append(j); values.append(1.0)
            for j in incoming.get(state, []):
                row_idx.append(current_row); col_idx.append(j); values.append(-1.0)
            rhs = 1.0 if state == source else -1.0 if state == sink else 0.0
            rhs_lb.append(rhs); rhs_ub.append(rhs)
            current_row += 1

    task_to_arc_indices = defaultdict(list)
    for j, arc in enumerate(all_arcs):
        if arc["kind"] == "task":
            task_to_arc_indices[arc["task"]].append(j)
    for task_id in tasks:
        for j in task_to_arc_indices[task_id]:
            row_idx.append(current_row); col_idx.append(j); values.append(1.0)
        rhs_lb.append(-np.inf); rhs_ub.append(1.0)
        current_row += 1

    A = coo_matrix(
        (values, (row_idx, col_idx)), shape=(current_row, n_vars)
    ).tocsr()
    constraints = LinearConstraint(
        A,
        np.asarray(rhs_lb, dtype=float),
        np.asarray(rhs_ub, dtype=float),
    )
    return c, integrality, bounds, constraints, A

def prepare_exact_problem(scenario, verbose=False):
    graph_start = time.perf_counter()
    states_by_h, arcs_by_h, all_arcs = build_full_exact_graph(scenario, verbose)
    graph_time = time.perf_counter() - graph_start

    matrix_start = time.perf_counter()
    c, integrality, bounds, constraints, A = build_exact_milp_matrices(
        scenario, states_by_h, arcs_by_h, all_arcs
    )
    matrix_time = time.perf_counter() - matrix_start
    return {
        "states_by_h": states_by_h,
        "arcs_by_h": arcs_by_h,
        "all_arcs": all_arcs,
        "c": c,
        "integrality": integrality,
        "bounds": bounds,
        "constraints": constraints,
        "A": A,
        "graph_time": graph_time,
        "matrix_time": matrix_time,
    }

def decode_exact_solution(scenario, result, all_arcs, method_name):
    if result.x is None:
        return pd.DataFrame()
    selected_arcs = [all_arcs[j] for j, value in enumerate(result.x) if value > 0.5]
    selected_by_h = defaultdict(list)
    for arc in selected_arcs:
        selected_by_h[arc["h"]].append(arc)

    rows = []
    for h in scenario["helicopters"]:
        outgoing = {arc["from"]: arc for arc in selected_by_h[h]}
        current = start_state(scenario, h)
        while True:
            arc = outgoing.get(current)
            if arc is None or arc["kind"] == "end":
                break
            if arc["kind"] == "task":
                task_id = arc["task"]
                task = scenario["tasks"][task_id]
                next_state = arc["to"]
                rows.append({
                    "Scenario": scenario["scenario_id"],
                    "Method": method_name,
                    "Helo ID": helo_to_int(h),
                    "Task ID": task_id,
                    "Task Class": task["priority_class"],
                    "Start Node": arc["from"][0],
                    "Node": scenario["task_to_node"][task_id],
                    "Start Time": arc["start_time"],
                    "Flight Time": arc["flight_time"],
                    "Service Time": arc["service_time"],
                    "Total Task Time": arc["flight_time"] + arc["service_time"],
                    "Cycle Time Proxy": np.nan,
                    "Completion Time": arc["finish_time"],
                    "Benefit": arc["benefit"],
                    "Selection Score": np.nan,
                    "Raw Impact": task["raw_impact"],
                    "Normalized Impact": task["impact"],
                    "Remaining Load": next_state[2],
                    "Remaining Medical": next_state[3],
                    "Remaining Personnel": next_state[4],
                    "Status": "Task Executed",
                })
            elif arc["kind"] == "return":
                next_state = arc["to"]
                rows.append({
                    "Scenario": scenario["scenario_id"],
                    "Method": method_name,
                    "Helo ID": helo_to_int(h),
                    "Task ID": "Idle",
                    "Task Class": "return",
                    "Start Node": arc["from"][0],
                    "Node": "Base",
                    "Start Time": arc["start_time"],
                    "Flight Time": arc["flight_time"],
                    "Service Time": arc["service_time"],
                    "Total Task Time": arc["flight_time"] + arc["service_time"],
                    "Cycle Time Proxy": np.nan,
                    "Completion Time": arc["finish_time"],
                    "Benefit": 0.0,
                    "Selection Score": np.nan,
                    "Raw Impact": 0.0,
                    "Normalized Impact": 0.0,
                    "Remaining Load": next_state[2],
                    "Remaining Medical": next_state[3],
                    "Remaining Personnel": next_state[4],
                    "Status": "Return/Recovery",
                })
            current = arc["to"]

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Completion Time", "Helo ID"]).reset_index(drop=True)
    return df

def validate_solution_consistency(scenario, df, method_name):
    if df is None or df.empty:
        return True
    task_df = df[df["Status"] == "Task Executed"].copy()
    if task_df["Task ID"].astype(int).duplicated().any():
        raise AssertionError(f"{method_name}: duplicated executed task.")
    if not task_df.empty and (task_df["Benefit"] <= 0).any():
        raise AssertionError(f"{method_name}: non-positive executed-task benefit.")
    if not task_df.empty:
        recalculated = sum(
            task_benefit(scenario, int(row["Task ID"]), float(row["Completion Time"]))
            for _, row in task_df.iterrows()
        )
        if not math.isclose(recalculated, task_df["Benefit"].sum(), abs_tol=1e-9):
            raise AssertionError(f"{method_name}: objective/log mismatch.")

    for helo_id in task_df["Helo ID"].unique():
        helo_rows = df[df["Helo ID"] == helo_id].sort_values("Completion Time")
        last = helo_rows.iloc[-1]
        if last["Node"] != "Base":
            raise AssertionError(f"{method_name}: H{helo_id} does not terminate at Base.")
        if last["Completion Time"] > scenario["params"]["TMAX"] + 1e-9:
            raise AssertionError(f"{method_name}: H{helo_id} exceeds TMAX.")
    return True

def empty_metric_row(scenario, summary):
    return {
        "Scenario": scenario["scenario_id"],
        "Seed": scenario["seed"],
        "Tasks": len(scenario["tasks"]),
        "Method": summary["Method"],
        "Normalized Mission Value": np.nan,
        "Weighted Impact Coverage": np.nan,
        "Impact Coverage": np.nan,
        "Physical Completed Tasks": np.nan,
        "Physical Completion Rate": np.nan,
        "Medical Evacuees": np.nan,
        "Personnel Transported": np.nan,
        "Cargo Delivered kg": np.nan,
        "Medical Impact Coverage": np.nan,
        "Personnel Impact Coverage": np.nan,
        "Cargo Impact Coverage": np.nan,
        "Average Response Time": np.nan,
        "Maximum Response Time": np.nan,
        "Makespan": np.nan,
        "Total Flight Time": np.nan,
        "Task Flight Time": np.nan,
        "Return Flight Time": np.nan,
        "Value per Flight Minute": np.nan,
        **summary,
    }

def compute_ttap_metrics(scenario, df, summary):
    if df is None or df.empty:
        row = empty_metric_row(scenario, summary)

        # No incumbent: metrics remain NaN.
        if not bool(summary.get("Has Incumbent", False)):
            return row

        # Valid incumbent with no executed task: zero-value plan.
        row.update({
            "Normalized Mission Value": 0.0,
            "Weighted Impact Coverage": 0.0,
            "Impact Coverage": 0.0,
            "Physical Completed Tasks": 0,
            "Physical Completion Rate": 0.0,
            "Medical Evacuees": 0.0,
            "Personnel Transported": 0.0,
            "Cargo Delivered kg": 0.0,
            "Medical Impact Coverage": 0.0,
            "Personnel Impact Coverage": 0.0,
            "Cargo Impact Coverage": 0.0,
            "Makespan": 0.0,
            "Total Flight Time": 0.0,
            "Task Flight Time": 0.0,
            "Return Flight Time": 0.0,
        })
        return row

    task_df = df[df["Status"] == "Task Executed"].copy()
    return_df = df[df["Status"] != "Task Executed"].copy()
    completed_ids = [int(value) for value in task_df["Task ID"]]
    completed_tasks = [scenario["tasks"][task_id] for task_id in completed_ids]
    total_tasks = list(scenario["tasks"].values())
    mission_value = float(task_df["Benefit"].sum())
    completed_impact = sum(task["impact"] for task in completed_tasks)
    total_impact = sum(task["impact"] for task in total_tasks)
    completed_potential = sum(task["potential_value"] for task in completed_tasks)

    def class_raw_total(task_list, task_class):
        return sum(
            task["raw_impact"]
            for task in task_list
            if task["priority_class"] == task_class
        )

    def class_coverage(task_class):
        denominator = class_raw_total(total_tasks, task_class)
        numerator = class_raw_total(completed_tasks, task_class)
        return numerator / denominator if denominator > 0 else np.nan

    total_flight = float(df["Flight Time"].sum())
    makespan = float(df["Completion Time"].max())
    return {
        "Scenario": scenario["scenario_id"],
        "Seed": scenario["seed"],
        "Tasks": len(total_tasks),
        "Method": summary["Method"],
        "Normalized Mission Value": mission_value,
        "Weighted Impact Coverage": completed_potential / scenario["total_potential_value"],
        "Impact Coverage": completed_impact / total_impact,
        "Physical Completed Tasks": len(completed_ids),
        "Physical Completion Rate": len(completed_ids) / len(total_tasks),
        "Medical Evacuees": class_raw_total(completed_tasks, "medical"),
        "Personnel Transported": class_raw_total(completed_tasks, "personnel"),
        "Cargo Delivered kg": class_raw_total(completed_tasks, "cargo"),
        "Medical Impact Coverage": class_coverage("medical"),
        "Personnel Impact Coverage": class_coverage("personnel"),
        "Cargo Impact Coverage": class_coverage("cargo"),
        "Average Response Time": float(task_df["Completion Time"].mean()),
        "Maximum Response Time": float(task_df["Completion Time"].max()),
        "Makespan": makespan,
        "Total Flight Time": total_flight,
        "Task Flight Time": float(task_df["Flight Time"].sum()),
        "Return Flight Time": float(return_df["Flight Time"].sum()) if not return_df.empty else 0.0,
        "Value per Flight Minute": mission_value / total_flight if total_flight > 0 else np.nan,
        **summary,
    }

def validate_exact_incumbent(result, summary, package, tol=1e-6):
    has_incumbent = bool(summary["Has Incumbent"])

    if not has_incumbent:
        if result.x is not None:
            raise AssertionError(
                f"{summary['Method']}: variables returned without "
                "a finite objective."
            )
        return True

    x = np.asarray(result.x, dtype=float)

    if len(x) != len(package["c"]):
        raise AssertionError(
            f"{summary['Method']}: incorrect solution dimension."
        )

    if not np.all(np.isfinite(x)):
        raise AssertionError(
            f"{summary['Method']}: non-finite decision variables."
        )

    if np.max(np.abs(x - np.rint(x))) > tol:
        raise AssertionError(
            f"{summary['Method']}: non-integral incumbent."
        )

    if (x < -tol).any() or (x > 1 + tol).any():
        raise AssertionError(
            f"{summary['Method']}: incumbent violates variable bounds."
        )

    lhs = np.asarray(package["A"] @ x).ravel()
    lb = np.asarray(package["constraints"].lb)
    ub = np.asarray(package["constraints"].ub)

    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)

    if (lhs[finite_lb] < lb[finite_lb] - tol).any():
        raise AssertionError(
            f"{summary['Method']}: lower constraint violation."
        )

    if (lhs[finite_ub] > ub[finite_ub] + tol).any():
        raise AssertionError(
            f"{summary['Method']}: upper constraint violation."
        )

    objective_from_x = float(package["c"] @ x)

    if not math.isclose(
        float(result.fun),
        objective_from_x,
        rel_tol=0,
        abs_tol=tol,
    ):
        raise AssertionError(
            f"{summary['Method']}: result.fun != c @ x."
        )

    return True
