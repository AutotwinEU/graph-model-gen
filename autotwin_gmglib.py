from typing import Any
from typing import Callable
import json
import neo4j
import os
import pandas
import time
import networkx
import numpy
import matplotlib.pyplot as mpyplot
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.backend_bases as mbackend

pandas.options.mode.copy_on_write = False
mpyplot.rcParams["pdf.fonttype"] = 42
mpyplot.rcParams["ps.fonttype"] = 42


###############################################################################
# Public functions                                                            #
###############################################################################


def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Configuration.
    """
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
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


def export_model(model: networkx.DiGraph, config: dict[str, Any]):
    """Export a graph model to a Neo4j database.

    Args:
        model: Graph model.
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
        model_id = session.execute_write(lambda t: _write_model(t, model))
    return model_id


def load_log(config: dict[str, Any]) -> pandas.DataFrame:
    """Load an event log from a CSV file.

    Args:
        config: Configuration.

    Returns:
        Event log.
    """
    columns = ["time", "station", "part", "type", "activity"]
    head = pandas.DataFrame(columns=columns)
    head.loc[-1] = {
        "time": 0.0, "station": None, "part": None, "type": None, "activity": None
    }
    path = os.path.join(config["work_path"], config["data"]["path"])
    column_mappings = config["data"]["mappings"]["column"]
    original_columns = column_mappings.keys()
    dtypes = {
        original_column: str
        for original_column, column in column_mappings.items()
        if column != "time"
    }
    body = pandas.read_csv(
        path, usecols=original_columns, dtype=dtypes, na_filter=False
    )
    body.rename(columns=column_mappings, inplace=True)
    if "type" not in body.columns:
        body.insert(body.columns.get_loc("part") + 1, "type", "UNKNOWN")
    activity_mappings = config["data"]["mappings"]["activity"]
    original_activities = activity_mappings.keys()
    indices_to_drop = body[~body["activity"].isin(original_activities)].index
    body.drop(index=indices_to_drop, inplace=True)
    body.reset_index(drop=True, inplace=True)
    body.replace(to_replace={"activity": activity_mappings}, inplace=True)
    if pandas.api.types.is_numeric_dtype(body["time"].dtype):
        body["time"] = body["time"].map(lambda t: pandas.to_datetime(t, unit="s"))
    elif pandas.api.types.is_string_dtype(body["time"].dtype):
        body["time"] = body["time"].map(lambda t: pandas.to_datetime(t))
    else:
        raise RuntimeError("Unsupported time format")
    body["time"] = body["time"].map(lambda t: t.timestamp())
    body.sort_values(by="time", inplace=True, kind="stable")
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
    model = networkx.DiGraph(name=config["name"], version=config["version"])
    station_sublogs, part_sublogs = _extract_sublogs(log)
    _normalize_activities(station_sublogs, part_sublogs, log)
    _mine_topology(model, station_sublogs, part_sublogs, log)
    _identify_operations(model, station_sublogs, part_sublogs)
    _mine_formulas(model, station_sublogs, part_sublogs, log)
    window = [-1, len(log) - 1]
    _reconstruct_states(model, station_sublogs, part_sublogs, log, window)
    _mine_capacities(model, log, window)
    _mine_processing_times(model, station_sublogs, part_sublogs, log, window, config)
    _mine_transfer_times(model, station_sublogs, part_sublogs, log, window, config)
    _mine_routing_probabilities(model, part_sublogs, window)
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
    layout: Callable[[networkx.DiGraph], dict[str, numpy.ndarray]]
    = lambda g: networkx.nx_agraph.graphviz_layout(g, prog="circo"),
):
    """Show a graph model in a figure window.

    Args:
        model: Graph model.
        layout: Layout function.
    """
    name = model.graph["name"]
    version = model.graph["version"]
    title = name + " " + version if version else name

    stations = list(model.nodes.keys())
    station_flows = dict()
    for station in stations:
        operation = model.nodes[station]["operation"]
        formulas = model.nodes[station]["formulas"]
        if operation in {"ORDINARY", "DETACH"}:
            flow = []
            for formula in formulas:
                type_ = next(iter(formula["input"].keys()))
                major = type_.split("_")[0]
                if major not in flow:
                    flow.append(major)
            flow.sort()
            flow = "|".join(flow)
        elif operation == "ATTACH":
            flow = []
            for formula in formulas:
                type_ = next(iter(formula["output"].keys()))
                major = type_.split("_")[0]
                if major not in flow:
                    flow.append(major)
            flow.sort()
            flow = "|".join(flow)
        else:
            flow = None
        station_flows[station] = flow

    connections = list(model.edges.keys())
    connection_flows = dict()
    for connection in connections:
        flow = []
        transfer_times = model.edges[connection]["transfer_times"]
        for type_ in transfer_times.keys():
            major = type_.split("_")[0]
            if major not in flow:
                flow.append(major)
        flow.sort()
        flow = "|".join(flow)
        connection_flows[connection] = flow

    flows = []
    for flow in station_flows.values():
        if flow not in flows:
            flows.append(flow)
    for flow in connection_flows.values():
        if flow not in flows:
            flows.append(flow)
    if None in flows:
        flows.remove(None)
    flows.sort()

    mpyplot.title(title, fontsize=10)
    pos = layout(model)
    cmap = mpyplot.get_cmap("gist_rainbow")
    white = mcolors.to_rgba_array("white")
    black = mcolors.to_rgba_array("black")
    colors = cmap(numpy.linspace(0.0, 1.0, len(flows)))
    flow_colors = dict()
    for x in range(len(flows)):
        flow_colors[flows[x]] = colors[x].reshape(1, -1)

    paths = []
    for station in stations:
        flow = station_flows[station]
        if flow is None:
            color = white
        else:
            color = flow_colors[flow]
        path = networkx.draw_networkx_nodes(
            model, pos, nodelist=[station], node_color=color, edgecolors=black
        )
        paths.append(path)

    patches = []
    for connection in connections:
        flow = connection_flows[connection]
        if flow is None:
            color = white
        else:
            color = flow_colors[flow]
        patch = networkx.draw_networkx_edges(
            model, pos, edgelist=[connection], edge_color=color, arrowsize=20
        )
        patch[0].set_edgecolor(black)
        patches.append(patch[0])

    networkx.draw_networkx_labels(model, pos, font_size=8)

    handles = []
    labels = []
    for flow in flows:
        color = flow_colors[flow]
        handles.append(mpatches.Rectangle((0, 0), 0, 0, color=color))
        labels.append(flow)
    mpyplot.legend(handles, labels, fontsize=8, title="Flow", title_fontsize=8)

    annotation = mpyplot.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-"},
        multialignment="left",
        bbox={"boxstyle": "round", "facecolor": "white"},
        fontsize=8,
        visible=False,
        zorder=6,
    )

    def handle_mouse_motion(event: mbackend.MouseEvent):
        """Handle a mouse motion in a figure window.

        Args:
            event: Mouse motion event.
        """
        is_inside = False
        text = ""
        for x_ in range(len(paths)):
            if is_inside:
                break
            is_inside, _ = paths[x_].contains(event)
            if is_inside:
                station_ = stations[x_]
                attributes = model.nodes[station_]
                annotation.xy = pos[station_]
                text += "Station: " + str(station_) + "\n"
                text += "Operation: "
                text += get_display_text(attributes["operation"], 1) + "\n"
                text += "Formulas: "
                text += get_display_text(attributes["formulas"], 1) + "\n"
                text += "Buffer Capacities: "
                text += get_display_text(attributes["buffer_capacities"], 1) + "\n"
                text += "Machine Capacity: "
                text += get_display_text(attributes["machine_capacity"], 1) + "\n"
                text += "Processing Times: "
                text += get_display_text(attributes["processing_times"], 1)
        for x_ in range(len(patches)):
            if is_inside:
                break
            is_inside, _ = patches[x_].contains(event, radius=5)
            if is_inside:
                connection_ = connections[x_]
                attributes = model.edges[connection_]
                origin_xy = pos[connection_[0]]
                destination_xy = pos[connection_[1]]
                annotation.xy = (
                    (origin_xy[0] + destination_xy[0]) / 2,
                    (origin_xy[1] + destination_xy[1]) / 2,
                )
                text += "Origin Station: " + str(connection_[0]) + "\n"
                text += "Destination Station: " + str(connection_[1]) + "\n"
                text += "Routing Probabilities: "
                text += get_display_text(attributes["routing_probabilities"], 1) + "\n"
                text += "Transfer Times: "
                text += get_display_text(attributes["transfer_times"], 1)
        if is_inside:
            point_xy = mpyplot.gca().transData.transform(annotation.xy)
            center_xy = mpyplot.gcf().transFigure.transform((0.5, 0.5))
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
            if annotation.get_visible():
                annotation.set_visible(False)

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
        if isinstance(value, list):
            text = ""
            for x_ in range(len(value)):
                x_ += 1
                y_ = value[x_ - 1]
                text += "\n" + "     " * level
                text += str(x_) + ": " + get_display_text(y_, level + 1)
        elif isinstance(value, dict):
            text = ""
            for x_, y_ in value.items():
                if x_ in display_names.keys():
                    x_ = display_names[x_]
                text += "\n" + "     " * level
                text += str(x_) + ": " + get_display_text(y_, level + 1)
        elif isinstance(value, float) or isinstance(value, numpy.floating):
            text = f"{value:.2f}"
        else:
            text = str(value)
        return text

    mpyplot.ion()
    mpyplot.connect("motion_notify_event", handle_mouse_motion)
    mpyplot.show(block=True)


###############################################################################
# Private functions                                                           #
###############################################################################


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
    interval = list(config["neo4j"]["interval"])
    for x in range(len(interval)):
        if isinstance(interval[x], (int, float)):
            pass
        elif isinstance(interval[x], str):
            interval[x] = "datetime('" + interval[x] + "')"
        else:
            raise RuntimeError("Unsupported time format")

    data = transaction.run(
        f"""
        MATCH (ev:Event)-[:OCCURRED_AT]->(st:Station:Ensemble)
        MATCH (ev)-[:EXECUTED_BY]->(ss:Sensor)
        MATCH (ev)-[:ACTS_ON]->(en:Entity)
        WHERE {interval[0]} <= ev.timestamp <= {interval[1]}
        CALL {{
            WITH ev, en
            MATCH (ev0)-[:DF_CONTROL_FLOW_ITEM*0..]->(ev)
            MATCH (ev0)-[:ACTS_ON]->(en)
            MATCH (ev0)-[:ASSOCIATE_TYPE]->(ent:EntityType)
            RETURN toInteger(split(elementId(ev0), ':')[2]) AS id0,
                   ev0.timestamp as time0, ent
            ORDER BY time0 DESC, id0 DESC
            LIMIT 1
        }}
        RETURN toInteger(split(elementId(ev), ':')[2]) AS id,
               ev.timestamp AS time, st.sysId AS station,
               st.type AS role, en.sysId AS part,
               ent.code AS type,
               CASE
                    WHEN ss.type IS NOT NULL AND ss.subType IS NOT NULL
                    THEN ss.type + '_' + ss.subType
                    WHEN ss.type IS NOT NULL AND ss.subType IS NULL
                    THEN ss.type
                    ELSE 'UNKNOWN'
               END AS activity
        ORDER BY time, id
        """
    ).data()

    x = 0
    while x < len(data):
        if data[x]["role"] == "source":
            for y in range(x - 1, -1, -1):
                if data[y]["station"] == data[x]["station"]:
                    data.insert(y + 1, data[x].copy())
                    data[y + 1]["time"] = data[y]["time"]
                    data[y + 1]["activity"] = "ENTER"
                    x += 1
                    break
        elif data[x]["role"] == "sink":
            data.insert(x + 1, data[x].copy())
            data[x + 1]["activity"] = "EXIT"
            x += 1
        x += 1
    columns = ["time", "station", "part", "type", "activity"]
    log = pandas.DataFrame.from_records(data, columns=columns)

    return log


def _write_model(
    transaction: neo4j.ManagedTransaction,
    model: networkx.DiGraph
) -> str:
    """Write a graph model to an SKG instance.

    Args:
        transaction: Write transaction.
        model: Graph model.

    Returns:
        Model ID.
    """
    model_name = model.graph["name"]
    model_version = model.graph["version"]
    model_id = transaction.run(
        f"""
        CREATE (gm:GraphModel:Instance)
        SET gm.name = '{model_name}',
            gm.version = '{model_version}'
        RETURN elementId(gm) AS eid
        """
    ).data()[0]["eid"]

    type_ids = dict()
    for type_ in model.graph["types"]:
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

        buffer_capacities = attributes["buffer_capacities"]
        for type_, capacity in buffer_capacities.items():
            buffer_id = transaction.run(
                f"""
                CREATE (bf:Entity:Resource:Station:Buffer)
                SET bf.capacity = {capacity}
                RETURN elementId(bf) AS eid
                """
            ).data()[0]["eid"]
            transaction.run(
                f"""
                MATCH (ent:EntityType) WHERE elementId(ent) = '{type_ids[type_]}'
                MATCH (bf:Buffer) WHERE elementId(bf) = '{buffer_id}'
                CREATE (ent)-[:OCCUPIES]->(bf)
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

        machine_capacity = attributes["machine_capacity"]
        machine_id = transaction.run(
            f"""
            CREATE (mc:Entity:Resource:Station:Machine)
            SET mc.capacity = {machine_capacity}
            RETURN elementId(mc) AS eid
            """
        ).data()[0]["eid"]
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
                SET fm.processingTimeMean = {processing_times[x]['mean']},
                    fm.processingTimeStd = {processing_times[x]['std']}
                RETURN elementId(fm) AS eid
                """
            ).data()[0]["eid"]

            for type_, cardinality in formulas[x]["input"].items():
                transaction.run(
                    f"""
                    MATCH (ent:EntityType) WHERE elementId(ent) = '{type_ids[type_]}'
                    MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                    CREATE (ent)-[in:INPUT]->(fm)
                    SET in.cardinality = {cardinality}
                    """
                )
            for type_, cardinality in formulas[x]["output"].items():
                transaction.run(
                    f"""
                    MATCH (fm:Formula) WHERE elementId(fm) = '{formula_id}'
                    MATCH (ent:EntityType) WHERE elementId(ent) = '{type_ids[type_]}'
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
                    rt.transferTimeMean = {transfer_times[type_]['mean']},
                    rt.transferTimeStd = {transfer_times[type_]['std']}
                RETURN elementId(rt) AS eid
                """
            ).data()[0]["eid"]
            transaction.run(
                f"""
                MATCH (ent:EntityType) WHERE elementId(ent) = '{type_ids[type_]}'
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
    """Extract the sublogs of parts and stations.

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
    types = log["type"].unique()
    types = types[pandas.notnull(types)]
    model.graph["types"] = list(types)

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
        model.nodes[station]["formulas"] = []
        model.nodes[station]["buffer_capacities"] = dict()
        model.nodes[station]["machine_capacity"] = 0
        model.nodes[station]["processing_times"] = []

    for connection in connections:
        model.edges[connection]["routing_probabilities"] = dict()
        model.edges[connection]["transfer_times"] = dict()


def _identify_operations(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
):
    """Identify the specific operation at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
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

    for station in stations:
        if input_frequencies[station] <= 0 or output_frequencies[station] <= 0:
            model.nodes[station]["operation"] = "ORDINARY"
            continue

        if input_frequencies[station] / output_frequencies[station] > 1.5:
            if cross_frequencies[station] / output_frequencies[station] < 0.5:
                model.nodes[station]["operation"] = "COMPOSE"
            else:
                model.nodes[station]["operation"] = "ATTACH"
        elif output_frequencies[station] / input_frequencies[station] > 1.5:
            if cross_frequencies[station] / input_frequencies[station] < 0.5:
                model.nodes[station]["operation"] = "DECOMPOSE"
            else:
                model.nodes[station]["operation"] = "DETACH"
        else:
            if cross_frequencies[station] / input_frequencies[station] < 0.5:
                model.nodes[station]["operation"] = "REPLACE"
            else:
                model.nodes[station]["operation"] = "ORDINARY"


def _mine_formulas(
    model: networkx.DiGraph,
    station_sublogs: dict[str, pandas.DataFrame],
    part_sublogs: dict[str, pandas.DataFrame],
    log: pandas.DataFrame,
):
    """Mine input-output formulas at each station.

    Args:
        model: Graph model.
        station_sublogs: Station sublogs.
        part_sublogs: Part sublogs.
        log: Event log.
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
        elif operation in {"REPLACE", "ATTACH", "COMPOSE"}:
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
            output = dict()
            for j in range(len(station_sublog) - 1, -1, -1):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "EXIT_AP":
                    output.clear()
                    output[type_] = 1
                elif activity == "ENTER_BP":
                    event["output"].update(output)
        else:
            input_ = dict()
            for j in range(len(station_sublog) - 1, -1, -1):
                event = station_sublog.iloc[j]
                type_ = event["type"]
                activity = event["activity"]
                if activity == "ENTER_BP":
                    input_.clear()
                    input_[type_] = 1
                elif activity == "EXIT_AP":
                    event["input"].update(input_)
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
        frequencies = []
        formula = None
        for j in range(len(sublog)):
            event = sublog.iloc[j]
            type_ = event["type"]
            activity = event["activity"]
            input_ = event["input"]
            output = event["output"]
            if activity == "ENTER_BP":
                if operation in {"DETACH", "DECOMPOSE"}:
                    formula = {"input": {type_: 1}, "output": dict()}
                    formula["output"].update(output)
            elif activity == "EXIT_AP":
                if operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}:
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
                if (
                    formula is not None
                    and len(formula["input"]) > 0
                    and len(formula["output"]) > 0
                ):
                    formulas.append(formula)
                    frequencies.append(1)
                    formula = None


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

    for station, sublog in station_sublogs.items():
        operation = model.nodes[station]["operation"]
        if operation in {"REPLACE", "ATTACH", "COMPOSE"}:
            for j in range(0, len(sublog)):
                event = sublog.iloc[j]
                activity = event["activity"]
                if activity == "EXIT_AP":
                    i = sublog.index[j]
                    if i > window[0]:
                        window[0] = i
                    break
        elif operation in {"DETACH", "DECOMPOSE"}:
            for j in range(len(sublog) - 1, -1, -1):
                event = sublog.iloc[j]
                activity = event["activity"]
                if activity == "ENTER_BP":
                    i = sublog.index[j]
                    if i < window[1]:
                        window[1] = i
                    break

    log["state"] = None
    for i in range(window[0], window[1]):
        log.at[i, "state"] = _create_state(model)
    for sublog in station_sublogs.values():
        sublog["state"] = None
        sublog.update(log["state"])
    for sublog in part_sublogs.values():
        sublog["state"] = None
        sublog.update(log["state"])

    previous_state = log.at[window[0], "state"]
    floor_state = _create_state(model)
    for i in range(window[0] + 1, window[1]):
        event = log.loc[i]
        station = event["station"]
        part = event["part"]
        type_ = event["type"]
        activity = event["activity"]
        input_ = event["input"]
        output = event["output"]
        state = event["state"]
        _copy_state(previous_state, state)
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
            for input_type in state[station]["M"].keys():
                floor_state[station]["M"][input_type] = min(
                    state[station]["M"][input_type],
                    floor_state[station]["M"][input_type],
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


def _create_state(model: networkx.DiGraph) -> dict[str, dict[str, dict[str, int]]]:
    """Create a new state.

    Args:
        model: Graph model.

    Returns:
        New state.
    """
    state = dict()
    for station in model.nodes.keys():
        state[station] = dict()
        operation = model.nodes[station]["operation"]
        formulas = model.nodes[station]["formulas"]
        for location in ["B", "M"]:
            state[station][location] = dict()
            if location == "B":
                for formula in formulas:
                    for type_ in formula["input"].keys():
                        state[station][location][type_] = 0
            else:
                if operation in {"ORDINARY", "REPLACE", "ATTACH", "COMPOSE"}:
                    for formula in formulas:
                        for type_ in formula["input"].keys():
                            state[station][location][type_] = 0
                else:
                    for formula in formulas:
                        for type_ in formula["output"].keys():
                            state[station][location][type_] = 0
    return state


def _copy_state(
    source_state: dict[str, dict[str, dict[str, int]]],
    target_state: dict[str, dict[str, dict[str, int]]],
):
    """Copy one state to another.

    Args:
        source_state: Source state.
        target_state: Target state.
    """
    for station in source_state.keys():
        for location in source_state[station].keys():
            for type_ in source_state[station][location].keys():
                target_state[station][location][type_] = (
                    source_state[station][location][type_]
                )


def _mine_capacities(model: networkx.DiGraph, log: pandas.DataFrame, window: list[int]):
    """Mine buffer and machine capacities at each station.

    Args:
        model: Graph model.
        log: Event log.
        window: Definite window.
    """
    ceiling_state = _create_state(model)
    for i in range(window[0], window[1]):
        state = log.at[i, "state"]
        for station in state.keys():
            for location in state[station].keys():
                for type_ in state[station][location].keys():
                    ceiling_state[station][location][type_] = max(
                        state[station][location][type_],
                        ceiling_state[station][location][type_],
                    )

    for station in model.nodes.keys():
        buffer_capacities = model.nodes[station]["buffer_capacities"]
        for type_, load in ceiling_state[station]["B"].items():
            buffer_capacities[type_] = load
        operation = model.nodes[station]["operation"]
        if operation == "ORDINARY":
            machine_capacity = max(ceiling_state[station]["M"].values())
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
    for station, station_sublog in station_sublogs.items():
        formulas = model.nodes[station]["formulas"]
        operation = model.nodes[station]["operation"]
        counts = [0 for _ in range(len(formulas))]
        means = [0.0 for _ in range(len(formulas))]
        tses = [0.0 for _ in range(len(formulas))]
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

            sample = exit_event["time"] - enter_event["time"]
            is_blocked = False
            if not model.nodes[station]["is_sink"]:
                exit_part = exit_event["part"]
                exit_type = exit_event["type"]
                part_sublog = part_sublogs[exit_part]
                i = exit_index
                j = part_sublog.index.get_loc(i)
                next_event = part_sublog.iloc[j + 1]
                next_station = next_event["station"]
                buffer_load = exit_event["state"][next_station]["B"][exit_type]
                buffer_capacity = (
                    model.nodes[next_station]["buffer_capacities"][exit_type]
                )
                if buffer_load >= buffer_capacity:
                    release_delay = config["model"]["delays"]["release"]
                    event = exit_event
                    while (
                        i > window[0]
                        and exit_event["time"] - event["time"] <= release_delay
                    ):
                        i -= 1
                        event = log.loc[i]
                        buffer_load = event["state"][next_station]["B"][exit_type]
                        if buffer_load >= buffer_capacity:
                            is_blocked = True
                            break

            if not is_blocked:
                input_ = exit_event["input"]
                output = enter_event["output"]
                for x in range(len(formulas)):
                    if (
                        formulas[x]["input"].keys() == input_.keys()
                        and formulas[x]["output"].keys() == output.keys()
                    ):
                        counts[x] += 1
                        last_mean = means[x]
                        means[x] = means[x] + (sample - means[x]) / counts[x]
                        tses[x] = tses[x] + (sample - last_mean) * (sample - means[x])
                        break

        processing_times = model.nodes[station]["processing_times"]
        for x in range(len(formulas)):
            processing_times.append(dict())
            processing_times[x]["mean"] = means[x]
            if counts[x] <= 1:
                processing_times[x]["std"] = 0.0
            else:
                processing_times[x]["std"] = (tses[x] / (counts[x] - 1)) ** 0.5


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
    counts = {connection: dict() for connection in connections}
    means = {connection: dict() for connection in connections}
    tses = {connection: dict() for connection in connections}
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
            if type_ not in counts[connection].keys():
                counts[connection][type_] = 0
            if type_ not in means[connection].keys():
                means[connection][type_] = 0.0
            if type_ not in tses[connection].keys():
                tses[connection][type_] = 0.0

            sample = enter_event["time"] - exit_event["time"]
            is_queued = False
            operation = model.nodes[enter_station]["operation"]
            seize_delay = config["model"]["delays"]["seize"]
            if operation == "ORDINARY":
                machine_load = enter_event["state"][enter_station]["M"][type_]
                machine_capacity = (
                    model.nodes[enter_station]["machine_capacity"]
                )
                if machine_load >= machine_capacity:
                    i_ = i
                    event = enter_event
                    while (
                        i_ > window[0]
                        and enter_event["time"] - event["time"] <= seize_delay
                    ):
                        i_ -= 1
                        event = log.loc[i_]
                        machine_load = event["state"][enter_station]["M"][type_]
                        if machine_load >= machine_capacity:
                            is_queued = True
                            break
            elif operation in {"REPLACE", "ATTACH", "COMPOSE"}:
                station_sublog = station_sublogs[enter_station]
                j_ = station_sublog.index.get_loc(i)
                i_ = i
                event = enter_event
                while (
                    j_ > 0
                    and i_ > window[0]
                    and enter_event["time"] - event["time"] <= seize_delay
                ):
                    j_ -= 1
                    i_ = station_sublog.index[j_]
                    event = station_sublog.iloc[j_]
                    if event["activity"] == "EXIT_AP":
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
                    and enter_event["time"] - event["time"] <= seize_delay
                ):
                    j_ -= 1
                    i_ = station_sublog.index[j_]
                    event = station_sublog.iloc[j_]
                    if event["activity"] == "ENTER_AP":
                        is_queued = True
                        break

            if not is_queued:
                counts[connection][type_] += 1
                last_mean = means[connection][type_]
                means[connection][type_] = (
                    means[connection][type_]
                    + (sample - means[connection][type_])
                    / counts[connection][type_]
                )
                tses[connection][type_] = (
                    tses[connection][type_]
                    + (sample - last_mean)
                    * (sample - means[connection][type_])
                )

    for connection in connections:
        transfer_times = model.edges[connection]["transfer_times"]
        for type_ in counts[connection].keys():
            transfer_times[type_] = dict()
            transfer_times[type_]["mean"] = means[connection][type_]
            if counts[connection][type_] <= 1:
                transfer_times[type_]["std"] = 0.0
            else:
                transfer_times[type_]["std"] = (
                    tses[connection][type_] / (counts[connection][type_] - 1)
                ) ** 0.5


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
