from array import array

import numpy as np


def _global_scan_values(maximum, num_points):
    return [float(value) for value in np.linspace(0.0, maximum, num_points)]


def _fine_scan_values(center, maximum, num_steps, step_size):
    return [
        float(value)
        for value in center + np.arange(-num_steps, num_steps + 1) * step_size
        if 0.0 <= value <= maximum
    ]


def _run_2d_scan(
    df_mc,
    root,
    signal_total,
    targets,
    thresholds,
    x_key,
    y_key,
    x_values,
    y_values,
    fixed_cuts,
    include_kinfit,
    stage,
    metric_mode,
):
    points = []
    actions = []

    for x_value in x_values:
        for y_value in y_values:
            cuts = {**fixed_cuts, x_key: x_value, y_key: y_value}
            selection = (
                "m1_vs_m2_cut(MassPi0NeutralFit1, MassPi0NeutralFit2, "
                f"{cuts['cut1']:.12g}, {cuts['cut2']:.12g})"
            )
            if include_kinfit:
                selection += (
                    f" && Chi2NeutralKinFit < {cuts['cut3']:.12g}"
                    f" && Chi2PMKinFit < {cuts['cut4']:.12g}"
                )

            filtered = df_mc.Filter(selection)
            signal_ptr = filtered.Filter("mctruth == 1").Count()
            background_ptr = filtered.Filter("mctruth != 1").Count()
            points.append({**cuts, "signal_ptr": signal_ptr, "background_ptr": background_ptr})
            actions.extend([signal_ptr, background_ptr])

    print(f"Uruchamiam {stage}: {len(points)} punktów 2D...")
    root.RDF.RunGraphs(actions)

    for point in points:
        n_signal = int(point.pop("signal_ptr").GetValue())
        n_background = int(point.pop("background_ptr").GetValue())
        total = n_signal + n_background
        efficiency = n_signal / signal_total
        purity = n_signal / total if total else 0.0
        significance = n_signal / np.sqrt(total) if total else 0.0
        target_distance = (
            (purity - targets["purity"]) ** 2
            + (efficiency - targets["efficiency"]) ** 2
        )
        point.update(
            {
                "n_signal": n_signal,
                "n_background": n_background,
                "efficiency": efficiency,
                "purity": purity,
                "significance": significance,
                "target_distance": target_distance,
                "valid": efficiency >= thresholds["efficiency"]
                and purity >= thresholds["purity"],
            }
        )
        point["metric"] = (
            significance if metric_mode == "significance" else target_distance
        )

    valid_points = [point for point in points if point["valid"]]
    if not valid_points:
        raise RuntimeError(f"Brak punktów spełniających ograniczenia w {stage}.")
    best_point = (
        max(valid_points, key=lambda point: point["metric"])
        if metric_mode == "significance"
        else min(valid_points, key=lambda point: point["metric"])
    )
    print(
        f"{stage}: {x_key}={best_point[x_key]:.6g}, "
        f"{y_key}={best_point[y_key]:.6g}; "
        f"eff={best_point['efficiency']:.4f}, pur={best_point['purity']:.4f}"
    )
    return points, best_point


def _write_scan_tree(root, root_file, points, stage):
    root_file.cd()
    tree = root.TTree(f"scan_{stage}", f"Wyniki skanu {stage}")
    double_fields = [
        "cut1",
        "cut2",
        "cut3",
        "cut4",
        "efficiency",
        "purity",
        "significance",
        "target_distance",
        "metric",
    ]
    buffers = {name: array("d", [0.0]) for name in double_fields}
    buffers.update(
        {
            "n_signal": array("q", [0]),
            "n_background": array("q", [0]),
            "valid": array("i", [0]),
        }
    )
    for name in double_fields:
        tree.Branch(name, buffers[name], f"{name}/D")
    tree.Branch("n_signal", buffers["n_signal"], "n_signal/L")
    tree.Branch("n_background", buffers["n_background"], "n_background/L")
    tree.Branch("valid", buffers["valid"], "valid/I")

    for point in points:
        for name in double_fields:
            buffers[name][0] = point[name]
        buffers["n_signal"][0] = point["n_signal"]
        buffers["n_background"][0] = point["n_background"]
        buffers["valid"][0] = int(point["valid"])
        tree.Fill()
    tree.Write()


def _write_scan_maps(root, root_file, points, best_point, x_key, y_key, x_values, y_values, stage):
    root_file.cd()
    map_specs = [
        ("efficiency", "Efficiency"),
        ("purity", "Purity"),
        ("significance", "Significance S/sqrt(S+B)"),
        ("target_distance", "Odległość od celu"),
    ]
    x_step = x_values[1] - x_values[0] if len(x_values) > 1 else max(abs(x_values[0]) * 0.1, 1.0)
    y_step = y_values[1] - y_values[0] if len(y_values) > 1 else max(abs(y_values[0]) * 0.1, 1.0)
    canvas = root.TCanvas(f"canvas_{stage}", f"Skan {stage}", 1200, 900)
    canvas.Divide(2, 2)
    keep_alive = []

    for pad, (field, title) in enumerate(map_specs, start=1):
        hist = root.TH2D(
            f"h2_{field}_{stage}",
            f"{title} ({stage});{x_key};{y_key}",
            len(x_values),
            x_values[0] - x_step / 2,
            x_values[-1] + x_step / 2,
            len(y_values),
            y_values[0] - y_step / 2,
            y_values[-1] + y_step / 2,
        )
        for point in points:
            hist.Fill(point[x_key], point[y_key], point[field])
        hist.Write()

        canvas.cd(pad)
        hist.Draw("COLZ")
        marker = root.TMarker(best_point[x_key], best_point[y_key], 29)
        marker.SetMarkerColor(root.kBlack)
        marker.SetMarkerSize(2.0)
        marker.Draw("same")
        keep_alive.extend([hist, marker])

    root_file.cd()
    canvas.Write()
    canvas.Close()


def run_kinfit_scan(df_mc, root, config):
    coarse_points = config["coarse_points_per_cut"]
    fine_steps = config["fine_num_steps"]
    fine_factor = config["fine_step_factor"]
    if config["final_metric_mode"] not in ("significance", "target_distance"):
        raise ValueError(f"Nieznany FINAL_METRIC_MODE: {config['final_metric_mode']}")
    if coarse_points < 2 or fine_steps < 1 or not 0 < fine_factor < 1:
        raise ValueError("Liczba punktów/kroków skanu lub fine_step_factor jest niepoprawny.")

    signal_total_ptr = df_mc.Filter("mctruth == 1").Count()
    signal_total = int(signal_total_ptr.GetValue())
    if signal_total <= 0:
        raise ValueError("Brak sygnału w df_MC_cache; nie można obliczyć efficiency.")

    range_maxima = {
        "cut1": 200.0,
        "cut2": 200.0,
        "cut3": 500.0,
        "cut4": 100.0,
    }
    if any(not np.isfinite(value) or value <= 0 for value in range_maxima.values()):
        raise ValueError(f"Nie można wyznaczyć dodatnich zakresów globalnego skanu: {range_maxima}")
    print(
        "Globalne zakresy progów (od zera): "
        + ", ".join(f"{key} <= {value:.6g}" for key, value in range_maxima.items())
    )
    coarse_steps = {
        key: maximum / (coarse_points - 1)
        for key, maximum in range_maxima.items()
    }
    targets = {"purity": config["target_purity_start"], "efficiency": config["target_efficiency_start"]}
    thresholds = {"purity": config["min_purity"], "efficiency": config["min_efficiency"]}

    values_1_coarse = {
        key: _global_scan_values(range_maxima[key], coarse_points)
        for key in ("cut1", "cut2")
    }
    results_1_coarse, best_1_coarse = _run_2d_scan(
        df_mc,
        root,
        signal_total,
        targets,
        thresholds,
        "cut1",
        "cut2",
        values_1_coarse["cut1"],
        values_1_coarse["cut2"],
        {"cut3": range_maxima["cut3"], "cut4": range_maxima["cut4"]},
        False,
        "Etap1_Coarse",
        "target_distance"
    )
    values_1_fine = {
        key: _fine_scan_values(
            best_1_coarse[key], range_maxima[key], fine_steps, coarse_steps[key] * fine_factor
        )
        for key in ("cut1", "cut2")
    }
    results_1_fine, best_1_fine = _run_2d_scan(
        df_mc,
        root,
        signal_total,
        targets,
        thresholds,
        "cut1",
        "cut2",
        values_1_fine["cut1"],
        values_1_fine["cut2"],
        {"cut3": range_maxima["cut3"], "cut4": range_maxima["cut4"]},
        False,
        "Etap1_Fine",
        "target_distance"
    )

    fixed_cuts = {"cut1": best_1_fine["cut1"], "cut2": best_1_fine["cut2"]}

    targets = {"purity": config["target_purity"], "efficiency": config["target_efficiency"]}
    thresholds = {"purity": config["min_purity"], "efficiency": config["min_efficiency"]}
    values_2_coarse = {
        key: _global_scan_values(range_maxima[key], coarse_points)
        for key in ("cut3", "cut4")
    }
    metric_mode = config["final_metric_mode"]
    results_2_coarse, best_2_coarse = _run_2d_scan(
        df_mc,
        root,
        signal_total,
        targets,
        thresholds,
        "cut3",
        "cut4",
        values_2_coarse["cut3"],
        values_2_coarse["cut4"],
        fixed_cuts,
        True,
        "Etap2_Coarse",
        metric_mode,
    )
    values_2_fine = {
        key: _fine_scan_values(
            best_2_coarse[key], range_maxima[key], fine_steps, coarse_steps[key] * fine_factor
        )
        for key in ("cut3", "cut4")
    }
    results_2_fine, best_2_fine = _run_2d_scan(
        df_mc,
        root,
        signal_total,
        targets,
        thresholds,
        "cut3",
        "cut4",
        values_2_fine["cut3"],
        values_2_fine["cut4"],
        fixed_cuts,
        True,
        "Etap2_Fine",
        metric_mode,
    )

    root_file = root.TFile.Open(config["output_file"], "RECREATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Nie można utworzyć pliku ROOT: {config['output_file']}")
    stages = [
        (results_1_coarse, best_1_coarse, "cut1", "cut2", values_1_coarse, "Etap1_Coarse"),
        (results_1_fine, best_1_fine, "cut1", "cut2", values_1_fine, "Etap1_Fine"),
        (results_2_coarse, best_2_coarse, "cut3", "cut4", values_2_coarse, "Etap2_Coarse"),
        (results_2_fine, best_2_fine, "cut3", "cut4", values_2_fine, "Etap2_Fine"),
    ]
    try:
        for points, best, x_key, y_key, values, stage in stages:
            _write_scan_tree(root, root_file, points, stage)
            _write_scan_maps(
                root, root_file, points, best, x_key, y_key, values[x_key], values[y_key], stage
            )
        _write_scan_tree(root, root_file, [best_2_fine], "BestPoint")
        root.TNamed("final_metric_mode", metric_mode).Write()
        for key, maximum in range_maxima.items():
            root.TNamed(f"scan_range_{key}", f"0 .. {maximum:.12g}").Write()
        scan_metadata = {
            "coarse_points_per_cut": coarse_points,
            "fine_num_steps": fine_steps,
            "fine_step_factor": fine_factor,
            "target_purity": targets["purity"],
            "target_efficiency": targets["efficiency"],
            "min_purity": thresholds["purity"],
            "min_efficiency": thresholds["efficiency"],
        }
        for key, value in scan_metadata.items():
            root.TNamed(f"scan_config_{key}", str(value)).Write()
    finally:
        root_file.Close()

    print(f"Wyniki, mapy 2D i canvasy zapisano do: {config['output_file']}")
    print(
        f"Finalne cięcia: cut1={best_2_fine['cut1']:.6g}, "
        f"cut2={best_2_fine['cut2']:.6g}, cut3={best_2_fine['cut3']:.6g}, "
        f"cut4={best_2_fine['cut4']:.6g}"
    )
    print(
        f"Efficiency={best_2_fine['efficiency']:.4f}, "
        f"Purity={best_2_fine['purity']:.4f}"
    )
    return {
        "results_step1_coarse": results_1_coarse,
        "results_step1_fine": results_1_fine,
        "results_step2_coarse": results_2_coarse,
        "results_step2_fine": results_2_fine,
        "best_step1_coarse": best_1_coarse,
        "best_step1_fine": best_1_fine,
        "best_step2_coarse": best_2_coarse,
        "best_step2_fine": best_2_fine,
        "best_cut1": best_1_fine["cut1"],
        "best_cut2": best_1_fine["cut2"],
        "best_cut3": best_2_fine["cut3"],
        "best_cut4": best_2_fine["cut4"],
        "best_eff_1": best_1_fine["efficiency"],
        "best_pur_1": best_1_fine["purity"],
        "best_eff_2b": best_2_fine["efficiency"],
        "best_pur_2b": best_2_fine["purity"],
    }
