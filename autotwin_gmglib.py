from typing import Any
from typing import Callable
import json
import neo4j
import os
import pandas
import datetime
import time
import networkx
import numpy
import matplotlib.pyplot as mpyplot
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.backend_bases as mbackend
import matplotlib.widgets as mwidgets
import threading
import warnings
import bisect
import scipy
import math

pandas.options.mode.copy_on_write = False
mpyplot.rcParams["pdf.fonttype"] = 42
mpyplot.rcParams["ps.fonttype"] = 42


###############################################################################
# Private constants                                                           #
###############################################################################
_LIBRARY_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_LIBRARY_FOLDER_PATH, "default.json")
_TIME_UNIT_FACTORS = {"h": 3600.0, "min": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6}


###############################################################################
# Public functions                                                            #
###############################################################################
def load_config(path: str = None) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Configuration.
    """
    with open(_DEFAULT_CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)

    if path is not None:
        with open(path, encoding="utf-8") as file:
            config = _deep_update(json.load(file), config)
        column_mappings = config["data"]["mappings"]["column"]
        columns_to_drop = set()
        for original_column, column in column_mappings.items():
            if original_column != column:
                columns_to_drop.add(column)
        for column in columns_to_drop:
            del column_mappings[column]
        activity_mappings = config["data"]["mappings"]["activity"]
        activities_to_drop = set()
        for original_activity, activity in activity_mappings.items():
            if original_activity != activity:
                activities_to_drop.add(activity)
        for activity in activities_to_drop:
            del activity_mappings[activity]

    return config


def import_log(config: dict[str, Any]):
    """Import an event log from a Neo4j database.

    Args:
        config: Configuration.
    """
    uri = config["neo4j"]["uri"]
    username = config["neo4j"]["username"]
    password = config["neo4j"]["password"]
    database = config["neo4j"]["database"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    with driver.session(database=database) as session:
        log = session.execute_read(lambda t: _read_log(t, config))
        path = os.path.join(config["work_path"], config["data"]["path"])
        log.to_csv(path, index=False)


def import_config(config: dict[str, Any]):
    """Import configuration from a Neo4j database.

    Args:
        config: Configuration.
    """
    uri = config["neo4j"]["uri"]
    username = config["neo4j"]["username"]
    password = config["neo4j"]["password"]
    database = config["neo4j"]["database"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    with driver.session(database=database) as session:
        session.execute_read(lambda t: _read_config(t, config))


def export_model(
    model: networkx.DiGraph,
    log: pandas.DataFrame,
    config: dict[str, Any],
):
    """Export a graph model to a Neo4j database.

    Args:
        model: Graph model.
        log: Event log.
        config: Configuration.

    Returns:
        Model ID.
    """
    uri = config["neo4j"]["uri"]
    username = config["neo4j"]["username"]
    password = config["neo4j"]["password"]
    database = config["neo4j"]["database"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    with driver.session(database=database) as session:
        model_id = session.execute_write(lambda t: _write_model(t, model, log))
    return model_id


def load_log(config: dict[str, Any]) -> pandas.DataFrame:
    """Load an event log from a CSV file.

    Args:
        config: Configuration.

    Returns:
        Event log.
    """
    columns = ["time", "unit", "station", "part", "family", "type", "activity", "npt"]
    head = pandas.DataFrame(columns=columns)
    head.loc[-1] = {column: (-1.0 if column == "time" else None) for column in columns}

    path = os.path.join(config["work_path"], config["data"]["path"])
    column_mappings = config["data"]["mappings"]["column"]
    original_columns = column_mappings.keys()
    dtypes = {
        original_column: str
        for original_column, column in column_mappings.items()
        if column != "time"
    }
    body = pandas.read_csv(
        path, usecols=lambda c: c in original_columns, dtype=dtypes, na_filter=False
    )
    body.rename(columns=column_mappings, inplace=True)
    if "unit" not in body.columns:
        body.insert(body.columns.get_loc("time") + 1, "unit", "")
    if "family" not in body.columns:
        body.insert(body.columns.get_loc("part") + 1, "family", "UNKNOWN")
    if "type" not in body.columns:
        body.insert(body.columns.get_loc("family") + 1, "type", "UNKNOWN")
    if "npt" not in body.columns:
        body.insert(body.columns.get_loc("activity") + 1, "npt", "")

    if config["data"]["clustering"]["path"] != "":
        clustering_path = os.path.join(
            config["work_path"], config["data"]["clustering"]["path"]
        )
        clustering_default = config["data"]["clustering"]["default"]
        clustering = pandas.read_csv(
            clustering_path, index_col="part", dtype=str, na_filter=False
        )
        body["family"] = "CLUSTER"
        for i in range(len(body)):
            part = body.at[i, "part"]
            if part in clustering.index:
                body.at[i, "type"] = clustering.at[part, "cluster"]
            else:
                body.at[i, "type"] = clustering_default
        indices_to_drop = body[body["type"] == ""].index
        body.drop(index=indices_to_drop, inplace=True)
        body.reset_index(drop=True, inplace=True)

    activity_mappings = config["data"]["mappings"]["activity"]
    original_activities = activity_mappings.keys()
    indices_to_drop = body[~body["activity"].isin(original_activities)].index
    body.drop(index=indices_to_drop, inplace=True)
    body.reset_index(drop=True, inplace=True)
    body.replace(to_replace={"activity": activity_mappings}, inplace=True)
    time_unit = config["model"]["time_unit"]
    body.sort_values(by="time", inplace=True, kind="stable")
    if pandas.api.types.is_numeric_dtype(body["time"].dtype):
        for i in range(len(body)):
            time_unit_ = "s" if body.at[i, "unit"] == "" else body.at[i, "unit"]
            time_ratio = _TIME_UNIT_FACTORS[time_unit_] / _TIME_UNIT_FACTORS[time_unit]
            body.at[i, "time"] *= time_ratio
            body.at[i, "unit"] = time_unit
    elif pandas.api.types.is_string_dtype(body["time"].dtype):
        workday = config["data"]["workday"]
        workday_start = datetime.time.fromisoformat(workday["start"])
        workday_end = datetime.time.fromisoformat(workday["end"])
        midnight = datetime.time.fromisoformat("00:00")
        previous_datetime = None
        time_ = 0.0
        time_ratio = _TIME_UNIT_FACTORS["s"] / _TIME_UNIT_FACTORS[time_unit]
        for i in range(len(body)):
            datetime_ = pandas.to_datetime(body.at[i, "time"])
            if workday_start != midnight and datetime_.time() < workday_start:
                datetime_ = datetime_.normalize().replace(
                    hour=workday_start.hour, minute=workday_start.minute
                )
            if workday_end != midnight and datetime_.time() > workday_end:
                datetime_ = datetime_.normalize().replace(
                    hour=workday_end.hour, minute=workday_end.minute
                )
            if (
                previous_datetime is not None
                and datetime_.date() == previous_datetime.date()
            ):
                time_ += (
                    datetime_.timestamp() - previous_datetime.timestamp()
                ) * time_ratio
            body.at[i, "time"] = time_
            body.at[i, "unit"] = time_unit
            previous_datetime = datetime_
    else:
        raise RuntimeError("Unsupported time format")
    for i in range(len(body)):
        npt = body.at[i, "npt"]
        if npt == "":
            npt = None
        else:
            npt = eval(npt.replace("inf", "float('inf')"))
            time_unit_ = "s" if npt["unit"] == "" else npt["unit"]
            time_ratio = _TIME_UNIT_FACTORS[time_unit_] / _TIME_UNIT_FACTORS[time_unit]
            npt["value"] *= time_ratio
            npt["min"] *= time_ratio
            npt["max"] *= time_ratio
        body.at[i, "npt"] = npt

    log = pandas.concat([head, body])
    return log


def generate_model(log: pandas.DataFrame, config: dict[str, Any]) -> networkx.DiGraph:
    """Generate a graph model from an event log.

    Args:
        log: Event log.
        config: Configuration.

    Returns:
        Graph model.
    """
    start_time = time.time()
    model = networkx.DiGraph(
        name=config["name"],
        version=config["version"],
        time_unit=config["model"]["time_unit"],
    )
    station_sublogs, part_sublogs = _extract_sublogs(log)
    _normalize_activities(station_sublogs, part_sublogs, log)
    _mine_topology(model, station_sublogs, part_sublogs, log)
    _identify_operations(model, station_sublogs, part_sublogs, config)
    _mine_formulas(model, station_sublogs, part_sublogs, log, config)
    window = [-1, len(log) - 1]
    _reconstruct_states(model, station_sublogs, part_sublogs, log, window)
    _mine_capacities(model, station_sublogs, log, window)
    _mine_processing_times(model, station_sublogs, part_sublogs, log, window, config)
    _mine_transfer_times(model, station_sublogs, part_sublogs, log, window, config)
    _mine_routing_probabilities(model, part_sublogs, window)
    _reduce_structure(model)
    model.graph["time_spent"] = time.time() - start_time
    return model


def save_model(model: networkx.DiGraph, config: dict[str, Any]):
    """Save a graph model as a JSON file.

    Args:
        model: Graph model.
        config: Configuration.
    """
    path = os.path.join(config["work_path"], config["model"]["path"])
    stream = networkx.readwrite.json_graph.adjacency_data(model)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(stream, file, indent=4)


def load_model(config: dict[str, Any]) -> networkx.DiGraph:
    """Load a graph model from a JSON file.

    Args:
        config: Configuration.

    Returns:
        Graph model.
    """
    path = os.path.join(config["work_path"], config["model"]["path"])
    with open(path, encoding="utf-8") as file:
        stream = json.load(file)
    model = networkx.readwrite.json_graph.adjacency_graph(stream)
    return model


def show_model(
    model: networkx.DiGraph,
    layout: Callable[[networkx.DiGraph], dict[str, numpy.ndarray]] = (
        lambda g: networkx.nx_agraph.graphviz_layout(g, prog="circo")
    ),
):
    """Show a graph model in interactive figures.

    Args:
        model: Graph model.
        layout: Layout function.
    """
    name = model.graph["name"]
    version = model.graph["version"]
    time_unit = model.graph["time_unit"]
    types = model.graph["types"]

    stations = list(model.nodes.keys())
    station_families = dict()
    for station in stations:
        operation = model.nodes[station]["operation"]
        formulas = model.nodes[station]["formulas"]
        station_family = None
        if operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}:
            for formula in formulas:
                formula_family = None
                output_type = next(iter(formula["output"].keys()))
                output_family = None
                for family in types.keys():
                    if output_type in types[family]:
                        output_family = family
                        break
                for input_type in formula["input"].keys():
                    input_family = None
                    for family in types.keys():
                        if input_type in types[family]:
                            input_family = family
                            break
                    if input_family == output_family:
                        formula_family = output_family
                        break
                if station_family is None:
                    station_family = formula_family
                else:
                    if formula_family != station_family:
                        station_family = None
                if station_family is None:
                    break
        else:
            for formula in formulas:
                formula_family = None
                input_type = next(iter(formula["input"].keys()))
                input_family = None
                for family in types.keys():
                    if input_type in types[family]:
                        input_family = family
                        break
                for output_type in formula["output"].keys():
                    output_family = None
                    for family in types.keys():
                        if output_type in types[family]:
                            output_family = family
                            break
                    if output_family == input_family:
                        formula_family = input_family
                        break
                if station_family is None:
                    station_family = formula_family
                else:
                    if formula_family != station_family:
                        station_family = None
                if station_family is None:
                    break
        station_families[station] = station_family

    connections = list(model.edges.keys())
    connection_families = dict()
    for connection in connections:
        routing_probabilities = model.edges[connection]["routing_probabilities"]
        connection_family = None
        for routing_type in routing_probabilities.keys():
            routing_family = None
            for family in types.keys():
                if routing_type in types[family]:
                    routing_family = family
                    break
            if connection_family is None:
                connection_family = routing_family
            else:
                if routing_family != connection_family:
                    connection_family = None
            if connection_family is None:
                break
        connection_families[connection] = connection_family

    figure, axes = mpyplot.subplots()
    window_title = "Graph Model"
    figure.canvas.manager.set_window_title(window_title)
    axes_title = name
    axes.set_title(axes_title, fontsize=10.0)
    pos = layout(model)
    cmap = mpyplot.get_cmap("gist_rainbow")
    white = mcolors.to_rgba_array("white")
    black = mcolors.to_rgba_array("black")
    colors = cmap(numpy.linspace(0.0, 1.0, len(types)))
    family_colors = dict()
    x = 0
    for family in types.keys():
        family_colors[family] = colors[x].reshape(1, -1)
        x += 1

    paths = list()
    for station in stations:
        station_family = station_families[station]
        if station_family is None:
            color = white
        else:
            color = family_colors[station_family]
        path = networkx.draw_networkx_nodes(
            model, pos, nodelist=[station], node_color=color, edgecolors=black
        )
        paths.append(path)

    patches = list()
    for connection in connections:
        connection_family = connection_families[connection]
        if connection_family is None:
            color = white
        else:
            color = family_colors[connection_family]
        patch = networkx.draw_networkx_edges(
            model, pos, edgelist=[connection], edge_color=color, arrowsize=20
        )
        patch[0].set_edgecolor(black)
        patches.append(patch[0])

    networkx.draw_networkx_labels(model, pos, font_size=8.0)

    dummy_handler = mpatches.Rectangle((0.0, 0.0), 0.0, 0.0, fill=False, linewidth=0.0)
    basic_handles = [dummy_handler, dummy_handler, dummy_handler]
    basic_labels = ["Name: " + name, "Version: " + version, "Time Unit: " + time_unit]
    basic_legend = mpyplot.legend(
        basic_handles,
        basic_labels,
        loc="upper left",
        fontsize=8,
        handlelength=0.0,
        handletextpad=0.0,
        title="Basic Information",
        title_fontsize=8,
    )
    axes.add_artist(basic_legend)

    family_handles = list()
    family_labels = list()
    for family in types.keys():
        color = family_colors[family]
        family_handles.append(mpatches.Rectangle((0.0, 0.0), 0.0, 0.0, color=color))
        family_labels.append(family + ": " + ", ".join(types[family]))
    family_legend = mpyplot.legend(
        family_handles,
        family_labels,
        loc="upper right",
        fontsize=8,
        title="Family of Types",
        title_fontsize=8,
    )
    axes.add_artist(family_legend)

    focus = None
    lock = threading.Lock()
    annotation = axes.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "linestyle": "--"},
        bbox={"boxstyle": "round", "edgecolor": "lightgray", "facecolor": "white"},
        fontsize=8.0,
        multialignment="left",
        visible=False,
        zorder=6.0,
    )
    cdf_figure_num = -1
    cdf_focus = None
    cdf_index = -1

    def handle_mouse_motion(event: mbackend.MouseEvent):
        """Handle a mouse motion in the main figure.

        Args:
            event: Mouse motion event.
        """
        nonlocal focus
        nonlocal cdf_figure_num
        nonlocal cdf_focus
        nonlocal cdf_index
        is_inside = False
        text = ""
        for x_ in range(len(paths)):
            if is_inside:
                break
            is_inside, _ = paths[x_].contains(event)
            if is_inside:
                station_ = stations[x_]
                with lock:
                    if focus == station_:
                        return
                    focus = station_
                attributes = model.nodes[station_]
                annotation.xy = pos[station_]
                text += "Station: " + station_ + "\n"
                text += "Operation: "
                text += get_display_text(attributes["operation"], 1) + "\n"
                text += "Formulas: "
                text += get_display_text(attributes["formulas"], 1) + "\n"
                text += "Buffer Loads: "
                text += get_display_text(attributes["buffer_loads"], 1) + "\n"
                text += "Buffer Capacities: "
                text += get_display_text(attributes["buffer_capacities"], 1) + "\n"
                text += "Machine Loads: "
                text += get_display_text(attributes["machine_loads"], 1) + "\n"
                text += "Machine Capacity: "
                text += get_display_text(attributes["machine_capacity"], 1) + "\n"
                text += "Processing Times: "
                text += get_display_text(attributes["processing_times"], 1)
                if not mpyplot.fignum_exists(cdf_figure_num):
                    cdf_figure_num = create_cdf_figure()
                cdf_axes = mpyplot.figure(num=cdf_figure_num).axes[0]
                if len(cdf_axes.lines) > 2:
                    cdf_axes.lines[2].remove()
                if len(attributes["processing_times"]) <= 0:
                    cdf_axes_title = (
                        "No Processing Time CDFs at Station " + station_
                    )
                    cdf_axes.set_title(cdf_axes_title, fontsize=10.0)
                    cdf_axes.set_xlim(left=-0.05, right=0.05)
                    cdf_axes.set_ylim(bottom=0.0, top=1.2)
                    cdf_focus = station_
                    cdf_index = -1
                else:
                    cdf_axes_title = (
                        "Processing Time CDF of Formula 1 at Station " + station_
                    )
                    cdf_axes.set_title(cdf_axes_title, fontsize=10.0)
                    cdf = attributes["processing_times"][0]["cdf"]
                    warnings.simplefilter("ignore", category=UserWarning)
                    cdf_axes.set_xlim(left=cdf[0][0], right=cdf[-1][0])
                    warnings.simplefilter("default", category=UserWarning)
                    cdf_axes.set_ylim(bottom=0.0, top=1.2)
                    cdf_axes.step(*zip(*cdf), where="post", color="tab:blue", linewidth=1.0)
                    cdf_focus = station_
                    cdf_index = 0
        for x_ in range(len(patches)):
            if is_inside:
                break
            is_inside, _ = patches[x_].contains(event, radius=5.0)
            if is_inside:
                connection_ = connections[x_]
                with lock:
                    if focus == connection_:
                        return
                    focus = connection_
                attributes = model.edges[connection_]
                origin_xy = pos[connection_[0]]
                destination_xy = pos[connection_[1]]
                annotation.xy = (
                    (origin_xy[0] + destination_xy[0]) / 2,
                    (origin_xy[1] + destination_xy[1]) / 2,
                )
                text += "Origin Station: " + connection_[0] + "\n"
                text += "Destination Station: " + connection_[1] + "\n"
                text += "Routing Probabilities: "
                text += get_display_text(attributes["routing_probabilities"], 1) + "\n"
                text += "Transfer Times: "
                text += get_display_text(attributes["transfer_times"], 1)
                if not mpyplot.fignum_exists(cdf_figure_num):
                    cdf_figure_num = create_cdf_figure()
                cdf_axes = mpyplot.figure(num=cdf_figure_num).axes[0]
                if len(cdf_axes.lines) > 2:
                    cdf_axes.lines[2].remove()
                if len(attributes["transfer_times"]) <= 0:
                    cdf_axes_title = (
                        "No Transfer Time CDFs on Connection ("
                        + connection_[0] + ", " + connection_[1] + ")"
                    )
                    cdf_axes.set_title(cdf_axes_title, fontsize=10.0)
                    cdf_axes.set_xlim(left=-0.05, right=0.05)
                    cdf_axes.set_ylim(bottom=0.0, top=1.2)
                    cdf_focus = connection_
                    cdf_index = -1
                else:
                    types = list(attributes["transfer_times"].keys())
                    cdf_axes_title = (
                        "Transfer Time CDF of Type " + types[0] + " on Connection ("
                        + connection_[0] + ", " + connection_[1] + ")"
                    )
                    cdf_axes.set_title(cdf_axes_title, fontsize=10.0)
                    cdf = attributes["transfer_times"][types[0]]["cdf"]
                    warnings.simplefilter("ignore", category=UserWarning)
                    cdf_axes.set_xlim(left=cdf[0][0], right=cdf[-1][0])
                    warnings.simplefilter("default", category=UserWarning)
                    cdf_axes.set_ylim(bottom=0.0, top=1.2)
                    cdf_axes.step(*zip(*cdf), where="post", color="tab:blue", linewidth=1.0)
                    cdf_focus = connection_
                    cdf_index = 0
        if is_inside:
            point_xy = axes.transData.transform(annotation.xy)
            center_xy = figure.transFigure.transform((0.5, 0.5))
            if point_xy[0] <= center_xy[0]:
                annotation.set(horizontalalignment="left")
            else:
                annotation.set(horizontalalignment="right")
            if point_xy[1] <= center_xy[1] - 30.0:
                annotation.xyann = (0.0, 30.0)
                annotation.set(verticalalignment="bottom")
            else:
                annotation.xyann = (0.0, -30.0)
                annotation.set(verticalalignment="top")
            annotation.set_text(text)
            annotation.set_visible(True)
        else:
            with lock:
                focus = None
            if annotation.get_visible():
                annotation.set_visible(False)

    ignored_keys = {"cdf"}
    display_names = {
        "input": "Input",
        "output": "Output",
        "mean": "Mean",
        "std": "Standard Deviation",
    }

    def get_display_text(value: Any, level: int) -> str:
        """Get the display text of an attribute value.

        Args:
            value: Attribute value.
            level: Indent level.

        Returns:
            Display text.
        """
        text = ""
        if isinstance(value, list):
            for x_ in range(len(value)):
                y_ = value[x_]
                text += "\n" + "     " * level
                text += str(x_ + 1) + ": " + get_display_text(y_, level + 1)
        elif isinstance(value, dict):
            for x_, y_ in value.items():
                if x_ in ignored_keys:
                    continue
                if x_ in display_names.keys():
                    x_ = display_names[x_]
                text += "\n" + "     " * level
                text += str(x_) + ": " + get_display_text(y_, level + 1)
        elif isinstance(value, float) or isinstance(value, numpy.floating):
            text += f"{value:.2f}"
        else:
            text += str(value)
        return text

    def create_cdf_figure():
        """Create a new CDF figure.

        Returns:
            int: Number of the CDF figure.
        """
        cdf_figure, cdf_axes = mpyplot.subplots()
        cdf_window_title = "Processing/Transfer Time CDF"
        cdf_figure.canvas.manager.set_window_title(cdf_window_title)
        cdf_axes.axhline(color="black", linestyle="--", linewidth=1.0, visible=False)
        cdf_axes.axvline(color="black", linestyle="--", linewidth=1.0, visible=False)
        cdf_axes.text(
            0.02,
            0.975,
            "",
            bbox={"boxstyle": "round", "edgecolor": "lightgray", "facecolor": "white"},
            fontsize=8.0,
            verticalalignment="top",
            transform=cdf_axes.transAxes,
        )
        cdf_figure.canvas.mpl_connect("motion_notify_event", handle_cdf_mouse_motion)
        previous_axes = cdf_axes.inset_axes([0.76, 0.9, 0.1, 0.075])
        previous_axes.button = mwidgets.Button(previous_axes, "Previous")
        previous_axes.button.label.set_fontsize(8.0)
        previous_axes.button.on_clicked(handle_cdf_button_click)
        next_axes = cdf_axes.inset_axes([0.88, 0.9, 0.1, 0.075])
        next_axes.button = mwidgets.Button(next_axes, "Next")
        next_axes.button.label.set_fontsize(8)
        next_axes.button.on_clicked(handle_cdf_button_click)
        return cdf_figure.number

    def handle_cdf_mouse_motion(event):
        """Handle a mouse motion in the CDF figure.

        Args:
            event (mbases.MouseEvent): Mouse motion event.
        """
        nonlocal cdf_index
        if cdf_index < 0:
            return
        cdf_axes = mpyplot.figure(num=cdf_figure_num).axes[0]
        if event.inaxes:
            cdf_xdata, cdf_ydata = cdf_axes.lines[2].get_data()
            if event.xdata <= cdf_xdata[0]:
                x_ = 0
            else:
                x_ = bisect.bisect_right(cdf_xdata, event.xdata)
                x_ = min(x_, len(cdf_xdata) - 1)
            cdf_axes.lines[0].set_ydata([cdf_ydata[x_]])
            cdf_axes.lines[1].set_xdata([cdf_xdata[x_]])
            cdf_axes.lines[0].set_visible(True)
            cdf_axes.lines[1].set_visible(True)
            text = ""
            text += f"Quantile: {cdf_xdata[x_]:.2f}\n"
            text += f"Probability: {cdf_ydata[x_]:.2f}"
            cdf_axes.texts[0].set_text(text)
            cdf_axes.texts[0].set_visible(True)
        else:
            if cdf_axes.lines[0].get_visible():
                cdf_axes.lines[0].set_visible(False)
            if cdf_axes.lines[1].get_visible():
                cdf_axes.lines[1].set_visible(False)
            if cdf_axes.texts[0].get_visible():
                cdf_axes.texts[0].set_visible(False)

    def handle_cdf_button_click(event):
        """Handle a button click on the CDF figure.

        Args:
            event (mbases.MouseEvent): Button click event.
        """
        nonlocal cdf_index
        if cdf_index < 0:
            return
        button = event.inaxes.button
        if cdf_focus in stations:
            processing_times = model.nodes[cdf_focus]["processing_times"]
            if button.label.get_text() == "Previous":
                if cdf_index <= 0:
                    return
                cdf_index -= 1
            else:
                if cdf_index >= len(processing_times) - 1:
                    return
                cdf_index += 1
            cdf_axes = mpyplot.figure(num=cdf_figure_num).axes[0]
            if len(cdf_axes.lines) > 2:
                cdf_axes.lines[2].remove()
            cdf = processing_times[cdf_index]["cdf"]
            cdf_axes_title = (
                "Processing Time CDF of Formula " + str(cdf_index + 1)
                + " at Station " + cdf_focus
            )
        else:
            transfer_times = model.edges[cdf_focus[0], cdf_focus[1]]["transfer_times"]
            types = list(transfer_times.keys())
            if button.label.get_text() == "Previous":
                if cdf_index <= 0:
                    return
                cdf_index -= 1
            else:
                if cdf_index >= len(types) - 1:
                    return
                cdf_index += 1
            cdf_axes = mpyplot.figure(num=cdf_figure_num).axes[0]
            if len(cdf_axes.lines) > 2:
                cdf_axes.lines[2].remove()
            cdf = transfer_times[types[cdf_index]]["cdf"]
            cdf_axes_title = (
                "Transfer Time CDF of Type " + types[cdf_index]
                + " on Connection (" + cdf_focus[0] + ", " + cdf_focus[1] + ")"
            )
        cdf_axes.set_title(cdf_axes_title, fontsize=10.0)
        warnings.simplefilter("ignore", category=UserWarning)
        cdf_axes.set_xlim(left=cdf[0][0], right=cdf[-1][0])
        warnings.simplefilter("default", category=UserWarning)
        cdf_axes.set_ylim(bottom=0.0, top=1.2)
        cdf_axes.step(*zip(*cdf), where="post", color="tab:blue", linewidth=1.0)

    figure.canvas.mpl_connect("motion_notify_event", handle_mouse_motion)
    mpyplot.ion()
    mpyplot.show(block=True)


###############################################################################
# Private functions                                                           #
###############################################################################
def _deep_copy(source: Any) -> Any:
    """Make a deep copy of a JSON-like object.

    Args:
        source: Source object.

    Returns:
        Target object.
    """
    if isinstance(source, list):
        target = list()
        for x in range(len(source)):
            target.append(_deep_copy(source[x]))
    elif isinstance(source, dict):
        target = dict()
        for x in source.keys():
            target[x] = _deep_copy(source[x])
    else:
        target = source
    return target


def _deep_update(source: Any, target: Any) -> Any:
    """Make a deep update of a JSON-like object.

    Args:
        source: Source object.
        target: Target object.

    Returns:
        Target object.
    """
    if isinstance(source, list) and isinstance(target, list):
        for x in range(len(source)):
            if x < len(target):
                target[x] = _deep_update(source[x], target[x])
            else:
                target.append(_deep_copy(source[x]))
    elif isinstance(source, dict) and isinstance(target, dict):
        for x in source.keys():
            if x in target.keys():
                target[x] = _deep_update(source[x], target[x])
            else:
                target[x] = _deep_copy(source[x])
    else:
        target = _deep_copy(source)
    return target


def _read_log(
    transaction: neo4j.ManagedTransaction,
    config: dict[str, Any],
) -> pandas.DataFrame:
    """Read an event log from an SKG instance.

    Args:
        transaction: Read transaction.
        config: Configuration.

    Returns:
        Event log.
    """
    interval = list(config["data"]["filters"]["interval"])
    for x in range(len(interval)):
        if isinstance(interval[x], (int, float)):
            pass
        elif isinstance(interval[x], str):
            interval[x] = "datetime('" + interval[x] + "')"
        else:
            raise RuntimeError("Unsupported time format")
    stations = config["data"]["filters"]["station"]
    if len(stations) <= 0:
        stations = transaction.run(
            """
            MATCH (st:Station:Ensemble)
            RETURN collect(st.sysId) AS stations
            """
        ).data()[0]["stations"]
    families = config["data"]["filters"]["family"]
    if len(families) <= 0:
        families = transaction.run(
            """
            MATCH (ent:EntityType)
            RETURN collect(DISTINCT ent.familyCode) AS families
            """
        ).data()[0]["families"]
    types = config["data"]["filters"]["type"]
    if len(types) <= 0:
        types = transaction.run(
            """
            MATCH (ent:EntityType)
            RETURN collect(ent.code) AS types
            """
        ).data()[0]["types"]

    event_records = transaction.run(
        f"""
        MATCH (ev:Event)
        WHERE ev.simulated IS NULL AND {interval[0]} <= ev.timestamp <= {interval[1]}
        MATCH (ev)-[:OCCURRED_AT]->(st:Station:Ensemble)
        MATCH (ev)-[:ACTS_ON]->(en:Entity)
        CALL {{
             WITH ev, en
             CALL apoc.when(
                  NOT EXISTS {{
                      MATCH (en)-[:IS_OF_TYPE]->(ent1:EntityType)
                      MATCH (en)-[:IS_OF_TYPE]->(ent2:EntityType)
                      WHERE ent1 <> ent2
                  }},
                  'MATCH (en)-[:IS_OF_TYPE]->(ent:EntityType)
                   RETURN ent',
                  'MATCH (ev0)-[:DF_CONTROL_FLOW_ITEM*0..]->(ev)
                   MATCH (ev0)-[:ACTS_ON]->(en)
                   MATCH (ev0)-[:ASSOCIATE_TYPE]->(ent:EntityType)
                   RETURN ent
                   ORDER BY ev0.timestamp DESC, id(ev0) DESC
                   LIMIT 1',
                  {{ev:ev, en:en}}
             ) YIELD value
             MATCH (ev)-[:EXECUTED_BY]->(ss:Sensor)
             RETURN value.ent AS ent, ss
             LIMIT 1
        }}
        WITH *
        WHERE st.sysId IN {stations} AND ent.familyCode IN {families}
              AND ent.code IN {types}
        RETURN elementId(ev) AS eid, ev.timestamp AS time,
               CASE
                    WHEN ev.time_unit IS NOT NULL
                    THEN ev.time_unit
                    ELSE ''
               END AS unit,
               st.sysId AS station, en.sysId AS part,
               ent.familyCode AS family, ent.code AS type,
               CASE
                    WHEN ss.type IS NOT NULL AND ss.subType IS NOT NULL
                    THEN ss.type + '_' + ss.subType
                    WHEN ss.type IS NOT NULL AND ss.subType IS NULL
                    THEN ss.type
                    ELSE 'UNKNOWN'
               END AS activity
        ORDER BY time, id(ev)
        """
    ).data()

    x = 0
    while x < len(event_records):
        x_ = x + 1
        while x_ < len(event_records):
            if event_records[x_]["time"] > event_records[x]["time"]:
                break
            x_ += 1
        for y in range(x, x_):
            y_ = y
            for z in range(y + 1, x_):
                if event_records[z]["part"] == event_records[y_]["part"]:
                    has_dfp = transaction.run(
                        f"""
                        MATCH (ev1:Event) WHERE elementId(ev1) = '{event_records[z]["eid"]}'
                        MATCH (ev2:Event) WHERE elementId(ev2) = '{event_records[y_]["eid"]}'
                        MATCH dfp = shortestPath((ev1)-[:DF_CONTROL_FLOW_ITEM*]->(ev2))
                        WITH collect(dfp) AS dfps
                        RETURN CASE WHEN size(dfps) <= 0 THEN FALSE ELSE TRUE END AS has_dfp
                        """
                    ).data()[0]["has_dfp"]
                else:
                    has_dfp = False
                if has_dfp:
                    y_ = z
            if y_ > y:
                temp = event_records[y]
                event_records[y] = event_records[y_]
                event_records[y_] = temp
        x = x_

    source_stations = set()
    sink_stations = set()
    part_event_records = dict()
    for event_record in event_records:
        station = event_record["station"]
        part = event_record["part"]
        source_stations.add(station)
        sink_stations.add(station)
        if part not in part_event_records.keys():
            part_event_records[part] = list()
        part_event_records[part].append(event_record)
    for part in part_event_records.keys():
        previous_station = None
        for event_record in part_event_records[part]:
            station = event_record["station"]
            activity = event_record["activity"]
            if activity == "ENTER":
                if previous_station is not None:
                    source_stations.discard(station)
                    sink_stations.discard(previous_station)
                    previous_station = None
            else:
                previous_station = station

    x = 0
    while x < len(event_records):
        station = event_records[x]["station"]
        part = event_records[x]["part"]
        activity = event_records[x]["activity"]
        if station in source_stations:
            if activity.startswith("EXIT"):
                activity = "EXIT"
                event_records[x]["activity"] = "EXIT"
            if activity == "ENTER":
                del event_records[x]
                x -= 1
            else:
                last_station = part_event_records[part][-1]["station"]
                if last_station == station:
                    del event_records[x]
                    x -= 1
        if station in sink_stations:
            if activity.startswith("EXIT"):
                activity = "EXIT"
                event_records[x]["activity"] = "EXIT"
            if activity == "EXIT":
                del event_records[x]
                x -= 1
            else:
                first_station = part_event_records[part][0]["station"]
                if first_station == station:
                    del event_records[x]
                    x -= 1
        x += 1

    x = 0
    while x < len(event_records):
        station = event_records[x]["station"]
        if station in source_stations:
            for y in range(x - 1, -1, -1):
                if event_records[y]["station"] == station:
                    event_records.insert(y + 1, event_records[x].copy())
                    event_records[y + 1]["time"] = event_records[y]["time"]
                    event_records[y + 1]["activity"] = "ENTER"
                    x += 1
                    break
        if station in sink_stations:
            event_records.insert(x + 1, event_records[x].copy())
            event_records[x + 1]["activity"] = "EXIT"
            x += 1
        x += 1

    min_usage = config["data"]["usage"]
    parts_to_drop = set()
    for part in part_event_records.keys():
        event_record = part_event_records[part][-1]
        station = event_record["station"]
        activity = event_record["activity"]
        if station not in sink_stations and activity.startswith("EXIT"):
            try:
                x = event_records.index(event_record)
            except ValueError:
                x = -1
            if 0 <= x < len(event_records) * min_usage:
                parts_to_drop.add(part)
    x = 0
    while x < len(event_records):
        part = event_records[x]["part"]
        if part in parts_to_drop:
            del event_records[x]
            x -= 1
        x += 1

    npt_records = transaction.run(
        """
        MATCH (en:Entity)-[:HAS]->(npt:NominalProcessingTime)-[:AT]->(st:Station:Ensemble)
        RETURN en.sysId AS part, st.sysId AS station, npt{.value, .min, .max, .unit} AS npt
        """
    ).data()
    npts = dict()
    for npt_record in npt_records:
        part = npt_record["part"]
        station = npt_record["station"]
        npt = npt_record["npt"]
        if npt["min"] is None:
            npt["min"] = 0.0
        if npt["max"] is None:
            npt["max"] = float("inf")
        if npt["unit"] is None:
            npt["unit"] = ""
        npts[part, station] = npt
    for event_record in event_records:
        station = event_record["station"]
        part = event_record["part"]
        if (part, station) in npts.keys():
            event_record["npt"] = npts[part, station]
        else:
            event_record["npt"] = None

    columns = ["time", "unit", "station", "part", "family", "type", "activity", "npt"]
    log = pandas.DataFrame.from_records(event_records, columns=columns)
    return log


def _read_config(
    transaction: neo4j.ManagedTransaction,
    config: dict[str, Any],
):
    """Read configuration from an SKG instance.

    Args:
        transaction: Read transaction.
        config: Configuration.
    """
    skg_global = transaction.run("MATCH (gl:Global) RETURN gl").data()[0]["gl"]
    config["data"]["workday"]["start"] = skg_global["startOfWorkDay"]
    config["data"]["workday"]["end"] = skg_global["endOfWorkDay"]


def _write_model(
    transaction: neo4j.ManagedTransaction,
    model: networkx.DiGraph,
    log: pandas.DataFrame,
) -> str:
    """Write a graph model to an SKG instance.

    Args:
        transaction: Write transaction.
        model: Graph model.
        log: Event log.

    Returns:
        Model ID.
    """
    model_name = model.graph["name"]
    model_version = model.graph["version"]
    model_time_unit = model.graph["time_unit"]
    model_types = model.graph["types"]
    model_id = transaction.run(
        f"""
        CREATE (gm:GraphModel:Instance)
        SET gm.name = '{model_name}',
            gm.version = '{model_version}'
        RETURN elementId(gm) AS eid
        """
    ).data()[0]["eid"]

    type_ids = dict()
    for family in model_types.keys():
        if "CLUSTER" in model_types.keys():
            for type_ in model_types["CLUSTER"]:
                type_ids[type_] = transaction.run(
                    f"""
                    CREATE (ent:EntityCluster)
                    SET ent.code = '{type_}',
                        ent.familyCode = 'CLUSTER'
                    RETURN elementId(ent) AS eid
                    """
                ).data()[0]["eid"]
                parts = log.loc[log["type"] == type_, "part"].unique()
                for part in parts:
                    transaction.run(
                        f"""
                        MATCH (en:Entity {{sysId: '{part}'}})
                        MATCH (ent:EntityCluster) WHERE elementId(ent) = '{type_ids[type_]}'
                        CREATE (en)-[:BELONGS_TO]->(ent)
                        """
                    )
                transaction.run(
                    f"""
                    MATCH (ent:EntityCluster) WHERE elementId(ent) = '{type_ids[type_]}'
                    MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
                    CREATE (ent)-[:PART_OF]->(gm)
                    """
                )
        else:
            for type_ in model_types[family]:
                type_ids[type_] = transaction.run(
                    f"""
                    MATCH (ent:EntityType {{code: '{type_}'}})
                    MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
                    CREATE (ent)-[:PART_OF]->(gm)
                    RETURN elementId(ent) AS eid
                    """
                ).data()[0]["eid"]

    for station, attributes in model.nodes.items():
        operation = attributes["operation"]
        station_id = transaction.run(
            f"""
            MATCH (st:Station:Ensemble {{sysId: '{station}'}})
            MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
            SET st.operation = '{operation}'
            CREATE (st)-[:PART_OF]->(gm)
            RETURN elementId(st) AS eid
            """
        ).data()[0]["eid"]

        buffer_loads = attributes["buffer_loads"]
        buffer_capacities = attributes["buffer_capacities"]
        for family, capacity in buffer_capacities.items():
            buffer_id = transaction.run(
                f"""
                CREATE (bf:Entity:Resource:Station:Buffer)
                SET bf.capacity = {capacity}
                RETURN elementId(bf) AS eid
                """
            ).data()[0]["eid"]
            for type_ in buffer_loads.keys():
                if type_ in model_types[family]:
                    transaction.run(
                        f"""
                        MATCH (ent:EntityType|EntityCluster)
                        WHERE elementId(ent) = '{type_ids[type_]}'
                        MATCH (bf:Buffer) WHERE elementId(bf) = '{buffer_id}'
                        CREATE (ent)-[oc:OCCUPIES]->(bf)
                        SET oc.load = {buffer_loads[type_]}
                        """
                    )
            transaction.run(
                f"""
                MATCH (bf:Buffer) WHERE elementId(bf) = '{buffer_id}'
                MATCH (st:Station:Ensemble) WHERE elementId(st) = '{station_id}'
                CREATE (bf)-[:BELONGS_TO]->(st)
                """
            )
            transaction.run(
                f"""
                MATCH (bf:Buffer) WHERE elementId(bf) = '{buffer_id}'
                MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
                CREATE (bf)-[:PART_OF]->(gm)
                """
            )

        machine_loads = attributes["machine_loads"]
        machine_capacity = attributes["machine_capacity"]
        machine_id = transaction.run(
            f"""
            CREATE (mc:Entity:Resource:Station:Machine)
            SET mc.capacity = {machine_capacity}
            RETURN elementId(mc) AS eid
            """
        ).data()[0]["eid"]
        for type_ in machine_loads.keys():
            transaction.run(
                f"""
                MATCH (ent:EntityType|EntityCluster)
                WHERE elementId(ent) = '{type_ids[type_]}'
                MATCH (mc:Machine) WHERE elementId(mc) = '{machine_id}'
                CREATE (ent)-[oc:OCCUPIES]->(mc)
                SET oc.load = {machine_loads[type_]}
                """
            )
        transaction.run(
            f"""
            MATCH (mc:Machine) WHERE elementId(mc) = '{machine_id}'
            MATCH (st:Station:Ensemble) WHERE elementId(st) = '{station_id}'
            CREATE (mc)-[:BELONGS_TO]->(st)
            """
        )
        transaction.run(
            f"""
            MATCH (mc:Machine) WHERE elementId(mc) = '{machine_id}'
            MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
            CREATE (mc)-[:PART_OF]->(gm)
            """
        )

        formulas = attributes["formulas"]
        processing_times = attributes["processing_times"]
        for x in range(len(formulas)):
            formula_id = transaction.run(
                f"""
                CREATE (fm:Entity:Resource:Station:Formula)
                SET fm.processingTimeMean = {processing_times[x]["mean"]},
                    fm.processingTimeStd = {processing_times[x]["std"]},
                    fm.processingTimeCdfX = {
                        [point[0] for point in processing_times[x]["cdf"]]
                    },
                    fm.processingTimeCdfY = {
                        [point[1] for point in processing_times[x]["cdf"]]
                    },
                    fm.processingTimeUnit = '{model_time_unit}'
                RETURN elementId(fm) AS eid
                """
            ).data()[0]["eid"]

            for type_, cardinality in formulas[x]["input"].items():
                transaction.run(
                    f"""
                    MATCH (ent:EntityType|EntityCluster)
                    WHERE elementId(ent) = '{type_ids[type_]}'
                    MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                    CREATE (ent)-[in:INPUT]->(fm)
                    SET in.cardinality = {cardinality}
                    """
                )
            for type_, cardinality in formulas[x]["output"].items():
                transaction.run(
                    f"""
                    MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                    MATCH (ent:EntityType|EntityCluster)
                    WHERE elementId(ent) = '{type_ids[type_]}'
                    CREATE (fm)-[ot:OUTPUT]->(ent)
                    SET ot.cardinality = {cardinality}
                    """
                )

            transaction.run(
                f"""
                MATCH (st:Station:Ensemble) WHERE elementId(st) = '{station_id}'
                MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                CREATE (st)-[:APPLIES]->(fm)
                """
            )
            transaction.run(
                f"""
                MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
                CREATE (fm)-[:PART_OF]->(gm)
                """
            )

    for connection, attributes in model.edges.items():
        connection_id = transaction.run(
            f"""
            MATCH (st1:Station:Ensemble {{sysId: '{connection[0]}'}})
                  -[:ORIGIN]->(cn:Connection:Ensemble)-[:DESTINATION]->
                  (st:Station:Ensemble {{sysId: '{connection[1]}'}})
            MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
            CREATE (cn)-[:PART_OF]->(gm)
            RETURN elementId(cn) AS eid
            """
        ).data()[0]["eid"]

        routing_probabilities = attributes["routing_probabilities"]
        transfer_times = attributes["transfer_times"]
        for type_ in routing_probabilities.keys():
            route_id = transaction.run(
                f"""
                CREATE (rt:Entity:Resource:Connection:Route)
                SET rt.probability = {routing_probabilities[type_]},
                    rt.transferTimeMean = {transfer_times[type_]["mean"]},
                    rt.transferTimeStd = {transfer_times[type_]["std"]},
                    rt.transferTimeCdfX = {
                        [point[0] for point in transfer_times[type_]["cdf"]]
                    },
                    rt.transferTimeCdfY = {
                        [point[1] for point in transfer_times[type_]["cdf"]]
                    },
                    rt.transferTimeUnit = '{model_time_unit}'
                RETURN elementId(rt) AS eid
                """
            ).data()[0]["eid"]
            transaction.run(
                f"""
                MATCH (ent:EntityType|EntityCluster)
                WHERE elementId(ent) = '{type_ids[type_]}'
                MATCH (rt:Route) WHERE elementId(rt) = '{route_id}'
                CREATE (ent)-[:OCCUPIES]->(rt)
                """
            )
            transaction.run(
                f"""
                MATCH (rt:Route) WHERE elementId(rt) = '{route_id}'
                MATCH (cn:Connection:Ensemble) WHERE elementId(cn) = '{connection_id}'
                CREATE (rt)-[:BELONGS_TO]->(cn)
                """
            )
            transaction.run(
                f"""
                MATCH (rt:Route) WHERE elementId(rt) = '{route_id}'
                MATCH (gm:GraphModel) WHERE elementId(gm) = '{model_id}'
                CREATE (rt)-[:PART_OF]->(gm)
                """
            )

    return model_id


def _extract_sublogs(
    log: pandas.DataFrame,
) -> tuple[dict[str, pandas.DataFrame], dict[str, pandas.DataFrame]]:
    """Extract the sublogs of stations and parts.

    Args:
        log: Event log.

    Returns:
        Tuple containing station sublogs and part sublogs.
    """
    stations = log["station"].unique()
    stations = stations[pandas.notnull(stations)]
    station_sublogs = dict()
    for station in stations:
        sublog = log[log["station"] == station].copy()
        station_sublogs[station] = sublog

    parts = log["part"].unique()
    parts = parts[pandas.notnull(parts)]
    part_sublogs = dict()
    for part in parts:
        sublog = log[log["part"] == part].copy()
        part_sublogs[part] = sublog

    return station_sublogs, part_sublogs


def _normalize_activities(
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
):
    """Normalize the activity for each event.

    Args:
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
    """
    log.loc[log["activity"] == "EXIT", "activity"] = "EXIT_AP"
    for sublog in station_sublogs.values():
        sublog.update(log["activity"])
    for sublog in part_sublogs.values():
        sublog.update(log["activity"])

    for i in range(len(log) - 1):
        event = log.loc[i]
        station = event["station"]
        part = event["part"]
        activity = event["activity"]
        if activity == "ENTER":
            station_sublog = station_sublogs[station]
            part_sublog = part_sublogs[part]
            j = part_sublog.index.get_loc(i)
            if j < len(part_sublog) - 1:
                next_event = part_sublog.iloc[j + 1]
                next_station = next_event["station"]
                next_activity = next_event["activity"]
                if next_station == station and next_activity == "EXIT_AR":
                    log.at[i, "activity"] = "ENTER_BR"
                    station_sublog.at[i, "activity"] = "ENTER_BR"
                    part_sublog.at[i, "activity"] = "ENTER_BR"
                    continue

            log.at[i, "activity"] = "ENTER_BP"
            station_sublog.at[i, "activity"] = "ENTER_BP"
            part_sublog.at[i, "activity"] = "ENTER_BP"


def _mine_topology(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
):
    """Mine the topology of the system.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
    """
    types = dict()
    for i in range(len(log) - 1):
        event = log.loc[i]
        family = event["family"]
        type_ = event["type"]
        if family is None or type_ is None:
            continue
        if family not in types.keys():
            types[family] = list()
        if type_ not in types[family]:
            types[family].append(type_)
    model.graph["types"] = types

    stations = station_sublogs.keys()
    for station in stations:
        model.add_node(station)

    connections = set()
    for sublog in part_sublogs.values():
        previous_station = None
        for j in range(len(sublog)):
            event = sublog.iloc[j]
            station = event["station"]
            activity = event["activity"]
            if activity.startswith("ENTER"):
                if previous_station is not None:
                    connection = (previous_station, station)
                    connections.add(connection)
                    previous_station = None
            else:
                previous_station = station
    for connection in connections:
        model.add_edge(*connection)

    for station in stations:
        model.nodes[station]["is_source"] = model.in_degree(station) <= 0
        model.nodes[station]["is_sink"] = model.out_degree(station) <= 0
        model.nodes[station]["operation"] = "ORDINARY"
        model.nodes[station]["formulas"] = list()
        model.nodes[station]["buffer_loads"] = dict()
        model.nodes[station]["buffer_capacities"] = dict()
        model.nodes[station]["machine_loads"] = dict()
        model.nodes[station]["machine_capacity"] = 0
        model.nodes[station]["processing_times"] = list()

    for connection in connections:
        model.edges[connection]["routing_probabilities"] = dict()
        model.edges[connection]["transfer_times"] = dict()


def _identify_operations(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    config: dict[str, Any],
):
    """Identify the specific operation at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        config: Configuration.
    """
    stations = model.nodes.keys()
    input_frequencies = dict.fromkeys(stations, 0)
    output_frequencies = dict.fromkeys(stations, 0)
    cross_frequencies = dict.fromkeys(stations, 0)
    for station in stations:
        station_sublog = station_sublogs[station]
        for j in range(len(station_sublog)):
            i = station_sublog.index[j]
            event = station_sublog.iloc[j]
            part = event["part"]
            activity = event["activity"]
            if activity == "ENTER_BP":
                input_frequencies[station] += 1
                part_sublog = part_sublogs[part]
                j_ = part_sublog.index.get_loc(i)
                if j_ < len(part_sublog) - 1:
                    next_event = part_sublog.iloc[j_ + 1]
                    next_station = next_event["station"]
                    next_activity = next_event["activity"]
                    if next_station == station and next_activity == "EXIT_AP":
                        cross_frequencies[station] += 1
            elif activity == "EXIT_AP":
                output_frequencies[station] += 1

    min_io_ratio = config["model"]["operation"]["io_ratio"]
    min_co_ratio = config["model"]["operation"]["co_ratio"]
    min_oi_ratio = config["model"]["operation"]["oi_ratio"]
    min_ci_ratio = config["model"]["operation"]["ci_ratio"]
    for station in stations:
        if input_frequencies[station] <= 0 or output_frequencies[station] <= 0:
            model.nodes[station]["operation"] = "ORDINARY"
            continue

        if input_frequencies[station] / output_frequencies[station] >= min_io_ratio:
            if cross_frequencies[station] / output_frequencies[station] >= min_co_ratio:
                model.nodes[station]["operation"] = "ATTACH"
            else:
                model.nodes[station]["operation"] = "COMPOSE"
        elif output_frequencies[station] / input_frequencies[station] >= min_oi_ratio:
            if cross_frequencies[station] / input_frequencies[station] >= min_ci_ratio:
                model.nodes[station]["operation"] = "DETACH"
            else:
                model.nodes[station]["operation"] = "DECOMPOSE"
        else:
            if (
                cross_frequencies[station] / output_frequencies[station] >= min_co_ratio
                and cross_frequencies[station] / input_frequencies[station] >= min_ci_ratio
            ):
                model.nodes[station]["operation"] = "ORDINARY"
            else:
                model.nodes[station]["operation"] = "REPLACE"


def _mine_formulas(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
    config: dict[str, Any],
):
    """Mine input-output formulas at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
        config: Configuration.
    """
    log["input"] = None
    log["output"] = None
    for i in range(len(log) - 1):
        event = log.loc[i]
        activity = event["activity"]
        if activity == "ENTER_BP":
            log.at[i, "output"] = dict()
        elif activity == "EXIT_AP":
            log.at[i, "input"] = dict()
    for sublog in station_sublogs.values():
        sublog["input"] = None
        sublog["output"] = None
        sublog.update(log["input"])
        sublog.update(log["output"])
    for sublog in part_sublogs.values():
        sublog["input"] = None
        sublog["output"] = None
        sublog.update(log["input"])
        sublog.update(log["output"])

    for station, station_sublog in station_sublogs.items():
        operation = model.nodes[station]["operation"]
        if operation == "ORDINARY":
            for j in range(len(station_sublog)):
                i = station_sublog.index[j]
                event = station_sublog.iloc[j]
                part = event["part"]
                activity = event["activity"]
                part_sublog = part_sublogs[part]
                j_ = part_sublog.index.get_loc(i)
                if activity == "ENTER_BP":
                    if j_ < len(part_sublog) - 1:
                        exit_event = part_sublog.iloc[j_ + 1]
                        output_type = exit_event["type"]
                        event["output"][output_type] = 1
                elif activity == "EXIT_AP":
                    if j_ > 0:
                        enter_event = part_sublog.iloc[j_ - 1]
                        input_type = enter_event["type"]
                        event["input"][input_type] = 1
        elif operation == "REPLACE":
            input_type = None
            for j in range(len(station_sublog)):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "ENTER_BP":
                    input_type = type_
                elif activity == "EXIT_AP":
                    if input_type is not None:
                        event["input"][input_type] = 1
                        input_type = None
            output_type = None
            for j in range(len(station_sublog) - 1, -1, -1):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "EXIT_AP":
                    output_type = type_
                elif activity == "ENTER_BP":
                    if output_type is not None:
                        event["output"][output_type] = 1
                        output_type = None
        elif operation in {"ATTACH", "COMPOSE"}:
            input_ = dict()
            for j in range(len(station_sublog)):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "ENTER_BP":
                    if type_ not in input_.keys():
                        input_[type_] = 0
                    input_[type_] += 1
                elif activity == "EXIT_AP":
                    event["input"].update(input_)
                    input_.clear()
            output_type = None
            for j in range(len(station_sublog) - 1, -1, -1):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "EXIT_AP":
                    output_type = type_
                elif activity == "ENTER_BP":
                    if output_type is not None:
                        event["output"][output_type] = 1
        else:
            input_type = None
            for j in range(len(station_sublog)):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "ENTER_BP":
                    input_type = type_
                elif activity == "EXIT_AP":
                    if input_type is not None:
                        event["input"][input_type] = 1
            output = dict()
            for j in range(len(station_sublog) - 1, -1, -1):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "EXIT_AP":
                    if type_ not in output.keys():
                        output[type_] = 0
                    output[type_] += 1
                elif activity == "ENTER_BP":
                    event["output"].update(output)
                    output.clear()

    for station, sublog in station_sublogs.items():
        operation = model.nodes[station]["operation"]
        formulas = model.nodes[station]["formulas"]
        frequencies = list()
        formula = None
        for j in range(len(sublog)):
            event = sublog.iloc[j]
            type_ = event["type"]
            activity = event["activity"]
            input_ = event["input"]
            output = event["output"]
            if activity == "ENTER_BP":
                if operation in {"DETACH", "DECOMPOSE"} and len(output) > 0:
                    formula = {"input": {type_: 1}, "output": dict()}
                    formula["output"].update(output)
            elif activity == "EXIT_AP":
                if (
                    operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}
                    and len(input_) > 0
                ):
                    formula = {"input": dict(), "output": {type_: 1}}
                    formula["input"].update(input_)

            if formula is not None:
                for x in range(len(formulas)):
                    if (
                        formulas[x]["input"].keys() == formula["input"].keys()
                        and formulas[x]["output"].keys() == formula["output"].keys()
                    ):
                        for type_ in formula["input"].keys():
                            formulas[x]["input"][type_] = max(
                                formula["input"][type_],
                                formulas[x]["input"][type_],
                            )
                        for type_ in formula["output"].keys():
                            formulas[x]["output"][type_] = max(
                                formula["output"][type_],
                                formulas[x]["output"][type_],
                            )
                        frequencies[x] += 1
                        formula = None
                        break
                if formula is not None:
                    formulas.append(formula)
                    frequencies.append(1)
                    formula = None

        if len(formulas) > 1:
            indexes = []
            max_frequency = max(frequencies)
            min_ratio = config["model"]["formula"]["ratio"]
            for x in range(len(frequencies)):
                if frequencies[x] / max_frequency < min_ratio:
                    indexes.append(x)
            indexes.reverse()
            for x in indexes:
                del formulas[x]


def _create_state(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
) -> dict[str, dict[str, dict[str, int]]]:
    """Create a new state.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.

    Returns:
        New state.
    """
    state = dict()
    for station, sublog in station_sublogs.items():
        operation = model.nodes[station]["operation"]
        state[station] = dict()
        state[station]["B"] = dict()
        state[station]["M"] = dict()
        for j in range(len(sublog)):
            event = sublog.iloc[j]
            type_ = event["type"]
            activity = event["activity"]
            if activity == "ENTER_BP":
                if operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}:
                    state[station]["B"][type_] = 0
                    state[station]["M"][type_] = 0
                else:
                    state[station]["B"][type_] = 0
            elif activity == "EXIT_AP":
                if operation in {"DETACH", "DECOMPOSE"}:
                    state[station]["M"][type_] = 0
            else:
                state[station]["B"][type_] = 0
                state[station]["M"][type_] = 0
    return state


def _reconstruct_states(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
    window: list[int],
):
    """Reconstruct the system state after each event.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
        window: Definite window.
    """
    for sublog in part_sublogs.values():
        j = len(sublog) - 1
        event = sublog.iloc[j]
        station = event["station"]
        activity = event["activity"]
        if not model.nodes[station]["is_sink"] and activity.startswith("EXIT"):
            i = sublog.index[j]
            if i < window[1]:
                window[1] = i

    zero_state = _create_state(model, station_sublogs)
    log["state"] = None
    for i in range(window[0], window[1]):
        log.at[i, "state"] = _deep_copy(zero_state)
    for sublog in station_sublogs.values():
        sublog["state"] = None
        sublog.update(log["state"])
    for sublog in part_sublogs.values():
        sublog["state"] = None
        sublog.update(log["state"])

    initial_state = log.at[window[0], "state"]
    previous_state = initial_state
    floor_state = _deep_copy(zero_state)
    for i in range(window[0] + 1, window[1]):
        event = log.loc[i]
        station = event["station"]
        part = event["part"]
        type_ = event["type"]
        activity = event["activity"]
        input_ = event["input"]
        output = event["output"]
        state = event["state"]
        state = _deep_update(previous_state, state)
        is_source = model.nodes[station]["is_source"]
        is_sink = model.nodes[station]["is_sink"]
        operation = model.nodes[station]["operation"]
        if activity.startswith("ENTER"):
            if not is_source:
                state[station]["B"][type_] -= 1
            if (
                activity == "ENTER_BP"
                and operation in {"DETACH", "DECOMPOSE"}
            ):
                for output_type, output_number in output.items():
                    state[station]["M"][output_type] += output_number
            else:
                state[station]["M"][type_] += 1
        else:
            if (
                activity == "EXIT_AP"
                and operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}
            ):
                for input_type, input_number in input_.items():
                    state[station]["M"][input_type] -= input_number
            else:
                state[station]["M"][type_] -= 1
            if not is_sink:
                sublog = part_sublogs[part]
                j = sublog.index.get_loc(i) + 1
                next_event = sublog.iloc[j]
                next_station = next_event["station"]
                state[next_station]["B"][type_] += 1
        if activity.startswith("ENTER"):
            floor_state[station]["B"][type_] = min(
                state[station]["B"][type_],
                floor_state[station]["B"][type_],
            )
        else:
            for type__ in state[station]["M"].keys():
                floor_state[station]["M"][type__] = min(
                    state[station]["M"][type__],
                    floor_state[station]["M"][type__],
                )
        previous_state = state

    for i in range(window[0], window[1]):
        state = log.at[i, "state"]
        for station in state.keys():
            for location in state[station].keys():
                for type_ in state[station][location].keys():
                    state[station][location][type_] -= (
                        floor_state[station][location][type_]
                    )
    if initial_state is None:
        for station in model.nodes.keys():
            model.nodes[station]["buffer_loads"].update(zero_state[station]["B"])
            model.nodes[station]["machine_loads"].update(zero_state[station]["M"])
    else:
        for station in model.nodes.keys():
            model.nodes[station]["buffer_loads"].update(initial_state[station]["B"])
            model.nodes[station]["machine_loads"].update(initial_state[station]["M"])


def _mine_capacities(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
    window: list[int],
):
    """Mine buffer and machine capacities at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        log: Event log.
        window: Definite window.
    """
    state = _create_state(model, station_sublogs)
    types = model.graph["types"]
    max_buffer_loads = dict()
    max_machine_loads = dict()
    for station in state.keys():
        max_buffer_loads[station] = dict()
        for type_ in state[station]["B"].keys():
            family = None
            for family in types.keys():
                if type_ in types[family]:
                    break
            max_buffer_loads[station][family] = 0
        max_machine_loads[station] = 0

    for i in range(window[0], window[1]):
        state = log.at[i, "state"]
        for station in state.keys():
            for family in max_buffer_loads[station].keys():
                buffer_load = 0
                for type_ in types[family]:
                    if type_ in state[station]["B"].keys():
                        buffer_load += state[station]["B"][type_]
                max_buffer_loads[station][family] = max(
                    buffer_load, max_buffer_loads[station][family]
                )
            station_load = sum(state[station]["M"].values())
            max_machine_loads[station] = max(
                station_load, max_machine_loads[station]
            )

    for station in model.nodes.keys():
        buffer_capacities = model.nodes[station]["buffer_capacities"]
        buffer_capacities.update(max_buffer_loads[station])
        operation = model.nodes[station]["operation"]
        if operation == "ORDINARY":
            machine_capacity = max_machine_loads[station]
        else:
            machine_capacity = 1
        model.nodes[station]["machine_capacity"] = machine_capacity


def _mine_processing_times(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
    window: list[int],
    config: dict[str, Any],
):
    """Mine processing times at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
        window: Definite window.
        config: Configuration.
    """
    types = model.graph["types"]
    replace_pts = config["model"]["cdf"]["replace_pts"]
    for station, station_sublog in station_sublogs.items():
        formulas = model.nodes[station]["formulas"]
        operation = model.nodes[station]["operation"]
        samples = [list() for _ in range(len(formulas))]
        for j in range(0, len(station_sublog)):
            i = station_sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue

            enter_event = None
            enter_index = -1
            exit_event = None
            exit_index = -1
            event = station_sublog.iloc[j]
            activity = event["activity"]
            npt = event["npt"]
            if activity == "ENTER_BP":
                enter_event = event
                enter_index = i
                if operation in {"DETACH", "DECOMPOSE"}:
                    for j_ in range(j + 1, len(station_sublog)):
                        next_event = station_sublog.iloc[j_]
                        next_activity = next_event["activity"]
                        if next_activity == "EXIT_AP":
                            i_ = station_sublog.index[j_]
                            if i_ < window[1]:
                                exit_event = station_sublog.iloc[j_]
                                exit_index = i_
                            break
            elif activity == "EXIT_AP":
                exit_event = event
                exit_index = i
                if operation == "ORDINARY":
                    exit_part = exit_event["part"]
                    part_sublog = part_sublogs[exit_part]
                    j_ = part_sublog.index.get_loc(i) - 1
                    if j_ > -1:
                        i_ = part_sublog.index[j_]
                        if i_ > window[0]:
                            enter_event = part_sublog.iloc[j_]
                            enter_index = i_
                elif operation in {"REPLACE", "ATTACH", "COMPOSE"}:
                    for j_ in range(j - 1, -1, -1):
                        previous_event = station_sublog.iloc[j_]
                        previous_activity = previous_event["activity"]
                        if previous_activity == "ENTER_BP":
                            i_ = station_sublog.index[j_]
                            if i_ > window[0]:
                                enter_event = station_sublog.iloc[j_]
                                enter_index = i_
                            break
            if (
                (enter_event is None and enter_index < 0)
                or (exit_event is None and exit_index < 0)
            ):
                continue

            is_blocked = False
            if not model.nodes[station]["is_sink"]:
                exit_part = exit_event["part"]
                exit_family = exit_event["family"]
                part_sublog = part_sublogs[exit_part]
                i = exit_index
                j = part_sublog.index.get_loc(i)
                next_event = part_sublog.iloc[j + 1]
                next_station = next_event["station"]
                buffer_load = 0
                for type_ in types[exit_family]:
                    if type_ in exit_event["state"][next_station]["B"].keys():
                        buffer_load += exit_event["state"][next_station]["B"][type_]
                buffer_capacity = (
                    model.nodes[next_station]["buffer_capacities"][exit_family]
                )
                if buffer_load >= buffer_capacity:
                    max_delay = config["model"]["delays"]["release"]
                    event = exit_event
                    while (
                        i > window[0]
                        and exit_event["time"] - event["time"] <= max_delay
                    ):
                        i -= 1
                        event = log.loc[i]
                        buffer_load = 0
                        for type_ in types[exit_family]:
                            if type_ in event["state"][next_station]["B"].keys():
                                buffer_load += event["state"][next_station]["B"][type_]
                        if buffer_load >= buffer_capacity:
                            is_blocked = True
                            break

            if is_blocked:
                sample = npt["value"] if npt is not None and replace_pts else None
            else:
                sample = float(exit_event["time"] - enter_event["time"])
                if npt is not None and (sample < npt["min"] or sample > npt["max"]):
                    sample = npt["value"] if replace_pts else None
            if sample is not None:
                input_ = exit_event["input"]
                output = enter_event["output"]
                for x in range(len(formulas)):
                    if (
                        formulas[x]["input"].keys() == input_.keys()
                        and formulas[x]["output"].keys() == output.keys()
                    ):
                        samples[x].append(sample)
                        break

        processing_times = model.nodes[station]["processing_times"]
        for x in range(len(formulas)):
            processing_times.append(dict())
            if len(samples[x]) <= 0:
                processing_times[x]["mean"] = 0.0
                processing_times[x]["std"] = 0.0
                processing_times[x]["cdf"] = [[0.0, 1.0]]
            elif len(samples[x]) == 1:
                processing_times[x]["mean"] = samples[x][0]
                processing_times[x]["std"] = 0.0
                processing_times[x]["cdf"] = [[samples[x][0], 1.0]]
            else:
                processing_times[x]["mean"] = float(scipy.stats.tmean(samples[x]))
                processing_times[x]["std"] = float(scipy.stats.tstd(samples[x]))
                processing_times[x]["cdf"] = _compute_empirical_cdf(samples[x], config)


def _mine_transfer_times(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
    window: list[int],
    config: dict[str, Any],
):
    """Mine transfer times on each connection.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
        window: Definite window.
        config: Configuration.
    """
    connections = model.edges.keys()
    samples = {connection: dict() for connection in connections}
    for part_sublog in part_sublogs.values():
        for j in range(0, len(part_sublog)):
            i = part_sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue

            enter_event = None
            enter_index = -1
            exit_event = None
            exit_index = -1
            event = part_sublog.iloc[j]
            activity = event["activity"]
            if activity.startswith("ENTER"):
                enter_event = event
                enter_index = i
                j_ = j - 1
                if j_ > -1:
                    i_ = part_sublog.index[j_]
                    if i_ > window[0]:
                        exit_event = part_sublog.iloc[j_]
                        exit_index = i_
            if (
                (enter_event is None and enter_index < 0)
                or (exit_event is None and exit_index < 0)
            ):
                continue

            enter_station = enter_event["station"]
            exit_station = exit_event["station"]
            connection = (exit_station, enter_station)
            type_ = event["type"]
            if type_ not in samples[connection].keys():
                samples[connection][type_] = list()

            sample = float(enter_event["time"] - exit_event["time"])
            is_queued = False
            operation = model.nodes[enter_station]["operation"]
            max_delay = config["model"]["delays"]["seize"]
            if operation == "ORDINARY":
                machine_load = enter_event["state"][enter_station]["M"][type_]
                machine_capacity = model.nodes[enter_station]["machine_capacity"]
                if machine_load >= machine_capacity:
                    i_ = i
                    event = enter_event
                    while (
                        i_ > window[0]
                        and enter_event["time"] - event["time"] <= max_delay
                    ):
                        i_ -= 1
                        event = log.loc[i_]
                        machine_load = event["state"][enter_station]["M"][type_]
                        if machine_load >= machine_capacity:
                            is_queued = True
                            break
            else:
                station_sublog = station_sublogs[enter_station]
                j_ = station_sublog.index.get_loc(i)
                i_ = i
                event = enter_event
                while (
                    j_ > 0
                    and i_ > window[0]
                    and enter_event["time"] - event["time"] <= max_delay
                ):
                    j_ -= 1
                    i_ = station_sublog.index[j_]
                    event = station_sublog.iloc[j_]
                    if event["activity"].startswith("EXIT"):
                        is_queued = True
                        break

            if not is_queued:
                samples[connection][type_].append(sample)

    for connection in connections:
        transfer_times = model.edges[connection]["transfer_times"]
        for type_ in samples[connection].keys():
            transfer_times[type_] = dict()
            if len(samples[connection][type_]) <= 0:
                transfer_times[type_]["mean"] = 0.0
                transfer_times[type_]["std"] = 0.0
                transfer_times[type_]["cdf"] = [[0.0, 1.0]]
            elif len(samples[connection][type_]) == 1:
                transfer_times[type_]["mean"] = samples[connection][type_][0]
                transfer_times[type_]["std"] = 0.0
                transfer_times[type_]["cdf"] = [[samples[connection][type_][0], 1.0]]
            else:
                transfer_times[type_]["mean"] = float(
                    scipy.stats.tmean(samples[connection][type_])
                )
                transfer_times[type_]["std"] = float(
                    scipy.stats.tstd(samples[connection][type_])
                )
                transfer_times[type_]["cdf"] = _compute_empirical_cdf(
                    samples[connection][type_], config
                )


def _compute_empirical_cdf(samples, config):
    """Compute the empirical CDF of multiple samples.

    Args:
        samples (list[float]): Input samples.
        config (dict[str, Any]): Configuration.

    Returns:
        list[list[float]]: Empirical CDF
    """
    cdf = scipy.stats.ecdf(samples).cdf
    max_points = config["model"]["cdf"]["points"]
    if max_points < 0 or len(cdf.quantiles) <= max_points:
        quantiles = cdf.quantiles
        probabilities = cdf.probabilities
    else:
        bin_size = math.ceil(len(cdf.quantiles) / max_points)
        num_bins = math.ceil(len(cdf.quantiles) / bin_size)
        quantiles = [0.0 for _ in range(num_bins)]
        probabilities = [0.0 for _ in range(num_bins)]
        quantile = 0.0
        max_delta = 0.0
        for x in range(len(cdf.quantiles)):
            if x <= 0:
                delta = cdf.probabilities[x]
            else:
                delta = cdf.probabilities[x] - cdf.probabilities[x - 1]
            if delta > max_delta:
                max_delta = delta
                quantile = cdf.quantiles[x]
            if x % bin_size >= bin_size - 1 or x >= len(cdf.quantiles) - 1:
                y = x // bin_size
                quantiles[y] = quantile
                probabilities[y] = cdf.probabilities[x]
                quantile = 0.0
                max_delta = 0.0
    cdf = [
        [float(quantiles[x]), float(probabilities[x])] for x in range(len(quantiles))
    ]
    return cdf


def _mine_routing_probabilities(
    model: networkx.DiGraph,
    part_sublogs: dict[str, pandas.DataFrame],
    window: list[int],
):
    """Mine routing probabilities on each connection.

    Args:
        model: Graph model.
        part_sublogs: Part sublogs.
        window: Definite window.
    """
    connections = model.edges.keys()
    stations = model.nodes.keys()
    connection_frequencies = {connection: dict() for connection in connections}
    station_frequencies = {station: dict() for station in stations}
    for sublog in part_sublogs.values():
        for j in range(1, len(sublog)):
            i = sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue

            enter_event = None
            enter_index = -1
            exit_event = None
            exit_index = -1
            event = sublog.iloc[j]
            activity = event["activity"]
            if activity.startswith("ENTER"):
                enter_event = event
                enter_index = i
                j_ = j - 1
                if j_ > -1:
                    i_ = sublog.index[j_]
                    if i_ > window[0]:
                        exit_event = sublog.iloc[j_]
                        exit_index = i_
            if (
                (enter_event is None and enter_index < 0)
                or (exit_event is None and exit_index < 0)
            ):
                continue

            enter_station = enter_event["station"]
            exit_station = exit_event["station"]
            connection = (exit_station, enter_station)
            type_ = event["type"]
            if type_ not in connection_frequencies[connection].keys():
                connection_frequencies[connection][type_] = 0
            connection_frequencies[connection][type_] += 1
            if type_ not in station_frequencies[exit_station].keys():
                station_frequencies[exit_station][type_] = 0
            station_frequencies[exit_station][type_] += 1

    for connection in model.edges.keys():
        routing_probabilities = model.edges[connection]["routing_probabilities"]
        for type_ in connection_frequencies[connection].keys():
            connection_frequency = connection_frequencies[connection][type_]
            station_frequency = station_frequencies[connection[0]][type_]
            routing_probability = connection_frequency / station_frequency
            routing_probabilities[type_] = routing_probability


def _reduce_structure(model: networkx.DiGraph):
    """Reduce the structure of the model.

    Args:
        model: Graph model.
    """
    stations = set(model.nodes.keys())
    for station in stations:
        formulas = model.nodes[station]["formulas"]
        processing_times = model.nodes[station]["processing_times"]
        if len(formulas) <= 0 or len(processing_times) <= 0:
            model.remove_node(station)

    connections = set(model.edges.keys())
    for connection in connections:
        routing_probabilities = model.edges[connection]["routing_probabilities"]
        transfer_times = model.edges[connection]["transfer_times"]
        if len(routing_probabilities) <= 0 or len(transfer_times) <= 0:
            model.remove_edge(*connection)
