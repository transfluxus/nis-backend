import logging
import math
import traceback
from typing import Dict, List, NoReturn

import lxml
import pandas as pd
import toposort
from lxml import etree

from nexinfosys import case_sensitive
from nexinfosys.command_generators import global_functions_extended
from nexinfosys.command_generators.parser_ast_evaluators import ast_evaluator, obtain_subset_of_processors, \
    get_adapted_case_dataframe_filter, obtain_processors
from nexinfosys.command_generators.parser_field_parsers import string_to_ast, indicator_expression, number_interval
from nexinfosys.common.constants import Scope
from nexinfosys.common.helper import create_dictionary, PartialRetrievalDictionary, \
    get_interfaces_and_weights_from_expression
from nexinfosys.ie_exports.xml_export import export_model_to_xml
from nexinfosys.model_services import get_case_study_registry_objects, State
from nexinfosys.models import CodeImmutable
from nexinfosys.models.musiasem_concepts import ProblemStatement, Parameter, FactorsRelationDirectedFlowObservation, \
    Processor, Indicator, MatrixIndicator, IndicatorCategories, Benchmark
from nexinfosys.models.statistical_datasets import Dataset, Dimension, CodeList
from nexinfosys.solving.graph import ResultDict, ConflictResolution, InterfaceNode, ResultKey, \
    evaluate_parameters_for_scenario, Computed


def compute_dataframe_sankey(results: ResultDict) -> pd.DataFrame:
    data: List[Dict] = []
    for result_key, node_floatcomputed_dict in results.items():
        if result_key.scope == Scope.Total and result_key.conflict != ConflictResolution.Dismissed:

            for node, float_computed in node_floatcomputed_dict.items():
                if float_computed.computed == Computed.Yes:
                    for interface_fullname, weight in get_interfaces_and_weights_from_expression(
                            float_computed.value.exp):
                        data.append(
                            {"Scenario": result_key.scenario,
                             "Period": result_key.period,
                             "OriginProcessor": interface_fullname.split(":")[0],
                             "OriginInterface": interface_fullname.split(":")[1],
                             "DestinationProcessor": node.processor_name,
                             "DestinationInterface": node.interface_name if node.interface_name else node.type + ":" + node.orientation,
                             "RelationType": float_computed.computation_source.name if float_computed.computation_source else None,
                             "Quantity": weight
                             }
                        )

    df = pd.DataFrame(data)
    df.set_index(
        ["Scenario", "Period", "OriginProcessor", "OriginInterface", "DestinationProcessor", "DestinationInterface"],
        inplace=True)
    return df.sort_index()


def export_solver_data(datasets, results, dynamic_scenario, state, global_parameters, problem_statement) -> NoReturn:
    def get_indicators_evaluation_order(indicators: List[Indicator]) -> List[Indicator]:
        # Prepare a dictionary of indicators with indicators they depend on
        indicators_dict = create_dictionary(data={i.name: i for i in indicators})
        indicator_dependencies = {}
        state = State()
        for indicator in indicators:
            ast = string_to_ast(indicator_expression,
                                indicator.formula if case_sensitive else indicator.formula.lower())
            issues = []
            dependencies = set()
            _, variables = ast_evaluator(ast, state, None, issues)
            for variable in variables:
                if variable in indicators_dict:
                    dependencies.add(indicators_dict[variable.name].name)
            indicator_dependencies[indicator.name] = dependencies
        # Find the order of evaluation
        return [indicators_dict[ind] for ind in toposort.toposort_flatten(indicator_dependencies)]

    glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

    data = {result_key.as_string_tuple() + node.full_key:
                {"RoegenType": node.roegen_type if node else "-",
                 "Value": float_computed.value.val,
                 "Computed": float_computed.computed.name,
                 "ComputationSource": float_computed.computation_source.name if float_computed.computation_source else None,
                 "Observer": float_computed.observer,
                 "Expression": str(float_computed.value.exp),
                 "Unit": node.unit if node else "-",
                 "Level": node.processor.attributes.get('level', '') if node else "-",
                 "System": node.system if node else "-",
                 "Subsystem": node.subsystem.name if node else "-",
                 "Sphere": node.sphere if node else "-"
                 }
            for result_key, node_floatcomputed_dict in results.items()
            for node, float_computed in node_floatcomputed_dict.items()}

    df = pd.DataFrame.from_dict(data, orient='index')

    # Round all values to 3 decimals
    # df = df.round(3)
    # Give a name to the dataframe indexes
    index_names = [f.title() for f in
                   ResultKey._fields] + InterfaceNode.full_key_labels()
    df.index.names = index_names

    # Sort the dataframe based on indexes. Not necessary, only done for debugging purposes.
    df = df.sort_index(level=index_names)

    # print(df)

    # Create Matrices for Sankey graphs
    ds_flow_values = prepare_sankey_dataset(glb_idx, df)
    dataframe_sankey = compute_dataframe_sankey(results)

    # Convert model to XML and to DOM tree. Used by XPath expressions (Matrices and Global Indicators)
    _, p_map = export_model_to_xml(glb_idx)  # p_map: {(processor_full_path_name, Processor), ...}
    dom_tree = etree.fromstring(_).getroottree()  # dom_tree: DOM against which an XQuery can be executed

    # Obtain Analysis objects: Indicators and Benchmarks
    indicators = glb_idx.get(Indicator.partial_key())
    matrix_indicators = glb_idx.get(MatrixIndicator.partial_key())
    benchmarks = glb_idx.get(Benchmark.partial_key())

    # Filter out conflicts and prepare for case insensitiveness
    # Filter: Conflict!='Dismissed' and remove the column
    df_without_conflicts = get_conflicts_filtered_dataframe(df)
    inplace_case_sensitiveness_dataframe(df_without_conflicts)

    # Calculate ScalarIndicators (Local and Global)
    # First, parse all indicators to find the evaluation order
    indicators_order = get_indicators_evaluation_order(indicators)
    # Then, evaluate the indicators in the order of evaluation
    df_local_indicators, df_global_indicators = calculate_scalar_indicators(
        indicators_order, dom_tree, p_map, df_without_conflicts, global_parameters, problem_statement, state
    )
    # df_local_indicators = calculate_local_scalar_indicators(indicators, dom_tree, p_map, df_without_conflicts, global_parameters, problem_statement, state)
    # df_global_indicators = calculate_global_scalar_indicators(indicators, dom_tree, p_map, df_without_conflicts, df_local_indicators, global_parameters, problem_statement, state)

    # Calculate benchmarks
    ds_benchmarks = calculate_local_benchmarks(df_local_indicators,
                                               indicators)  # Find local indicators, and related benchmarks (indic_to_benchmarks). For each group (scenario, time, scope, processor): for each indicator, frame the related benchmark and add the framing result
    ds_global_benchmarks = calculate_global_benchmarks(df_global_indicators,
                                                       indicators)  # Find global indicators, and related benchmarks (indic_to_benchmarks). For each group (scenario, time, scope, processor): for each indicator, frame the related benchmark and add the framing result

    # Prepare Benchmarks to Stakeholders DataFrame
    ds_stakeholders = prepare_benchmarks_to_stakeholders(
        benchmarks)  # Find all benchmarks. For each benchmark, create a row per stakeholder -> return the dataframe

    # Prepare Indicator Matrices
    # TODO df_attributes
    matrices = prepare_matrix_indicators(matrix_indicators, glb_idx, dom_tree, p_map, df, df_local_indicators,
                                         dynamic_scenario)

    #
    # ---------------------- CREATE DATASETS AND STORE IN STATE ----------------------
    #

    if not dynamic_scenario:
        ds_name = "flow_graph_solution"
        ds_flows_name = "flow_graph_solution_edges"
        ds_sankey_name = "flow_graph_solution_sankey"
        ds_indicators_name = "flow_graph_solution_indicators"
        df_global_indicators_name = "flow_graph_global_indicators"
        ds_benchmarks_name = "flow_graph_solution_benchmarks"
        ds_global_benchmarks_name = "flow_graph_solution_global_benchmarks"
        ds_stakeholders_name = "benchmarks_and_stakeholders"
    else:
        ds_name = "dyn_flow_graph_solution"
        ds_flows_name = "dyn_flow_graph_solution_edges"
        ds_sankey_name = "dyn_flow_graph_solution_sankey"
        ds_indicators_name = "dyn_flow_graph_solution_indicators"
        df_global_indicators_name = "dyn_flow_graph_global_indicators"
        ds_benchmarks_name = "dyn_flow_graph_solution_benchmarks"
        ds_global_benchmarks_name = "dyn_flow_graph_solution_global_benchmarks"
        ds_stakeholders_name = "benchmarks_and_stakeholders"

    for d, name, label in [(df, ds_name, "Flow Graph Solver - Interfaces"),
                           (ds_flow_values, ds_flows_name, "Flow Graph Solver Edges - Interfaces"),
                           (dataframe_sankey, ds_sankey_name, "Flow Graph Solution - Sankey"),
                           (df_local_indicators, ds_indicators_name, "Flow Graph Solver - Local Indicators"),
                           (df_global_indicators, df_global_indicators_name, "Flow Graph Solver - Global Indicators"),
                           (ds_benchmarks, ds_benchmarks_name, "Flow Graph Solver - Local Benchmarks"),
                           (ds_global_benchmarks, ds_global_benchmarks_name, "Flow Graph Solver - Global Benchmarks"),
                           (ds_stakeholders, ds_stakeholders_name, "Benchmarks - Stakeholders")
                           ]:
        if not d.empty:
            datasets[name] = get_dataset(d, name, label)

    # Register matrices
    for n, ds in matrices.items():
        datasets[n] = ds

    # Create dataset and store in State (specific of "Biofuel case study")
    # datasets["end_use_matrix"] = get_eum_dataset(df)

    return []


def prepare_benchmarks_to_stakeholders(benchmarks: List[Benchmark]):
    rows = []
    for b in benchmarks:
        for s in b.stakeholders:
            rows.append((b.name, s))

    df = pd.DataFrame(data=rows, columns=["Benchmark", "Stakeholder"])
    df.set_index("Benchmark", inplace=True)
    return df


def add_conflicts_to_results(existing_results: ResultDict, taken_results: ResultDict, dismissed_results: ResultDict,
                             conflict_type: str) -> ResultDict:
    """ Iterate on the existing results and mark which of them have been involved into a conflict """
    results: ResultDict = {}
    for result_key, node_floatcomputed_dict in existing_results.items():

        if result_key in taken_results:
            assert result_key in dismissed_results
            key_taken = result_key._replace(**{conflict_type: ConflictResolution.Taken})
            key_dismissed = result_key._replace(**{conflict_type: ConflictResolution.Dismissed})

            for node, float_computed in node_floatcomputed_dict.items():
                if node in taken_results[result_key]:
                    results.setdefault(key_taken, {})[node] = taken_results[result_key][node]
                    results.setdefault(key_dismissed, {})[node] = dismissed_results[result_key][node]
                else:
                    results.setdefault(result_key, {})[node] = float_computed
        else:
            results[result_key] = node_floatcomputed_dict

    return results


def prepare_sankey_dataset(registry: PartialRetrievalDictionary, df: pd.DataFrame):
    # Create Matrix to Sankey graph

    FactorsRelationDirectedFlowObservation_list = registry.get(FactorsRelationDirectedFlowObservation.partial_key())

    ds_flows = pd.DataFrame({'source': [i._source.full_name for i in FactorsRelationDirectedFlowObservation_list],
                             'source_processor': [i._source._processor._name for i in
                                                  FactorsRelationDirectedFlowObservation_list],
                             'source_level': [i._source._processor._attributes['level'] if (
                                     'level' in i._source._processor._attributes) else None for i in
                                              FactorsRelationDirectedFlowObservation_list],
                             'target': [i._target.full_name for i in FactorsRelationDirectedFlowObservation_list],
                             'target_processor': [i._target._processor._name for i in
                                                  FactorsRelationDirectedFlowObservation_list],
                             'target_level': [i._target._processor._attributes[
                                                  'level'] if 'level' in i._target._processor._attributes else None for
                                              i in FactorsRelationDirectedFlowObservation_list],
                             # 'RoegenType_target': [i.target_factor._attributes['roegen_type']for i in FactorsRelationDirectedFlowObservation_list],
                             'Sphere_target': [i.target_factor._attributes['sphere'] for i in
                                               FactorsRelationDirectedFlowObservation_list],
                             'Subsystem_target': [i._target._processor._attributes['subsystem_type'] for i in
                                                  FactorsRelationDirectedFlowObservation_list],
                             'System_target': [i._target._processor._attributes['processor_system'] for i in
                                               FactorsRelationDirectedFlowObservation_list]
                             }
                            )

    # I suppose that relations between processors (source-target) doesn't change between different scenarios.
    df2 = df.reset_index()
    processor = df2["Processor"].apply(lambda x: x.split("."))
    df2["lastprocessor"] = [i[-1] for i in processor]
    df2["source"] = df2["lastprocessor"] + ":" + df2["Interface"]
    # df2 = df2[df2["Orientation"]=="Output"] It is not necessary?

    ds_flow_values = pd.merge(df2, ds_flows, on="source")
    ds_flow_values = ds_flow_values.drop(
        columns=["Orientation", "lastprocessor", "Processor", "Interface", 'RoegenType'], axis=1)
    ds_flow_values = ds_flow_values.rename(
        columns={'Sphere': 'Sphere_source', 'System': 'System_source', 'Subsystem': 'Subsystem_source'})
    # ds_flow_values.reset_index()
    # if not ds_flows.empty:

    return ds_flow_values


def get_conflicts_filtered_dataframe(in_df: pd.DataFrame) -> pd.DataFrame:
    filt = in_df.index.get_level_values("Conflict").isin(["No", "Taken"])
    df = in_df[filt]
    df = df.droplevel("Conflict")
    return df


def inplace_case_sensitiveness_dataframe(df: pd.DataFrame):
    if not case_sensitive:
        level_processor = df.index._get_level_number("Processor")
        level_interface = df.index._get_level_number("Interface")
        df.index = df.index.set_levels([df.index.levels[level_processor].str.lower(),
                                        df.index.levels[level_interface].str.lower()],
                                       level=[level_processor, level_interface],
                                       verify_integrity=False)


def calculate_scalar_indicators(indicators: List[Indicator],
                                serialized_model: lxml.etree._ElementTree,
                                p_map: Dict[str, Processor],
                                results: pd.DataFrame,
                                global_parameters: List[Parameter], problem_statement: ProblemStatement,
                                global_state: State) -> pd.DataFrame:
    """
    Compute local and global scalar indicators using data from "results", and return a pd.DataFrame for local
    indicators and another for global indicators

    :param indicators: List of indicators to compute
    :param serialized_model:
    :param p_map:
    :param results: Result of the graph solving process ("flow_graph_solution")
    :param global_parameters: List of parameter definitions
    :param problem_statement: Object with a list of scenarios (defining Parameter sets)
    :param global_state: Whole model State (used to obtain the PartialRetrievalDictionary)
    :return: (pd.DataFrame, pd.DataFrame) with all the local and global indicators
    """

    # The "columns" in the index of "results" are:
    # 'Scenario', 'System', 'Period', 'Scope', 'Processor', 'Interface', 'Orientation'
    # Group by: 'Scenario', 'System', 'Period', 'Scope', 'Processor'
    # Rearrange: 'Interface' and 'Orientation'
    local_idx_names = ["Scenario", "System", "Period", "Scope",
                       "Processor"]  # Changing factors. If this changes, update indices below
    global_idx_names = ["Scenario", "Period"]  # "System", "Scope"

    def select_processor_by_name(processor_names, processor_name: str) -> bool:
        if processor_names is None or len(processor_names) == 0:
            return True
        else:
            return processor_name in processor_names

    def prepare_state(processor_indicators, global_indicators, scenario_params, registry,
                      scenario, system, period, scope, processor_name, group, account_nas):
        # Find processor(s)
        # TODO SYSTEM ?? Processor.partial_key(processor_name, system=system)
        processor = registry.get(Processor.partial_key(processor_name))[0]
        # Processor attributes
        d = {k: v for k, v in processor.custom_attributes().items()}
        # Interface values
        ifaces = group.index.get_level_values(4).values if case_sensitive \
            else group.index.get_level_values(4).str.lower().values
        orient = group.index.get_level_values(5).values if case_sensitive \
            else group.index.get_level_values(5).str.lower().values
        values = group["Value"].values
        if case_sensitive:
            d.update({k: v for k, v in zip(ifaces, values)})
            d.update({f"{k}_{k2}": v for k, k2, v in zip(ifaces, orient, values)})
        else:
            d.update({k.lower(): v for k, v in zip(ifaces, values)})
            d.update({f"{k}_{k2}".lower(): v for k, k2, v in zip(ifaces, orient, values)})
        if account_nas:
            # Add interfaces which are currently not in the dictionary
            # (declared but with no value: "not available".
            #  If not declared they are "not applicable" (does not apply))
            for iface in processor.factors:
                if case_sensitive:
                    if iface.name not in d:
                        d[iface.name] = math.nan
                else:
                    if iface.name.lower() not in d:
                        d[iface.name.lower()] = math.nan
        # Local Indicator values
        d.update(processor_indicators.get((processor, scenario, system, period, scope), {}))
        # Parameters (they have priority over previous values, except global indicators)
        d.update(scenario_params.get(scenario, {}))
        # Global Indicator values
        d.update(global_indicators.get(system, {}))
        d.update(global_indicators.get("all", {}))
        return d, processor

    def calculate_local_scalar_indicator(indicator: Indicator) -> pd.DataFrame:
        """

        :param indicator:
        :param account_nas:
        :return:
        """
        account_nas = indicator._account_na  # If True, account NAvs, NAps and N, for each calculated indicator

        df = results

        # Parse the expression
        ast = string_to_ast(indicator_expression, indicator.formula if case_sensitive else indicator.formula.lower())

        # Scenario parameters
        scenario_params = create_dictionary()
        for scenario_name, scenario_exp_params in problem_statement.scenarios.items():  # type: str, dict
            scenario_params[scenario_name] = evaluate_parameters_for_scenario(global_parameters, scenario_exp_params)
            if not case_sensitive:
                scenario_params[scenario_name] = {k.lower(): v for k, v in scenario_params[scenario_name].items()}

        issues = []
        new_df_rows_idx = []
        new_df_rows_data = []

        # Obtain subset of processors, if "processors_selector" is defined
        if indicator.processors_selector:
            inv_map = {v: k for k, v in p_map.items()}
            processors = obtain_processors(indicator.processors_selector, serialized_model, p_map)
            processor_names = set([inv_map[p] for p in processors])
        else:
            processor_names = None

        for t, g in df.groupby(local_idx_names):  # "t", tuple characterizing the group; "g", values in the group
            scenario = t[0]
            system = t[1]
            period = t[2]
            scope = t[3]
            processor_name = t[4]
            # Check if it is wanted to calculate the indicator for the processor
            if not select_processor_by_name(processor_names, processor_name):
                continue
            d, proc = prepare_state(processor_indicators, global_indicators, scenario_params, registry,
                                    scenario, system, period, scope, processor_name,
                                    group=g, account_nas=account_nas)

            state = State(d)
            # LCIA Methods, for the special "lciamethods" function
            state.set("_lcia_methods", global_state.get("_lcia_methods"))
            if account_nas:
                # TODO If an interface is declared, put a np.NaN, at this point
                # TODO If an interface is not declared, assume zero. How to differentiate between parameter and
                #      interface, in Scalar Indicator expressions?
                # TODO Add all Processor interfaces with an np.NaN
                pass

            # Evaluate!!
            val, variables = ast_evaluator(ast, state, None, issues,
                                           account_nas_name=indicator.name if account_nas else None)
            p_t = (proc, scenario, system, period, scope)
            # Gather results
            if val is not None:  # If it was possible to evaluate ... append a new row
                if isinstance(val, dict):  # LCIA method returns a Dict, or any formula when "account_nas=True"
                    # Two options for the naming of resulting indicators:
                    # 1. If only one LCIA indicator, use the name in the column
                    # 2. If more than one LCIA indicator, use the auto-generated name
                    if (len(val) == 4 and account_nas) or (len(val) == 1 and not account_nas):
                        def _get_indicator_name(indicator_name, k):
                            if k.lower().endswith("_nap"):
                                return f"{indicator_name}_NAp"
                            elif k.lower().endswith("_nav"):
                                return f"{indicator_name}_NAv"
                            elif k.lower().endswith("_n"):
                                return f"{indicator_name}_N"
                            else:
                                return indicator_name

                        for k, v in val.items():
                            l = list(t)
                            i_name = _get_indicator_name(indicator.name, k)
                            l.append(i_name)
                            t2 = tuple(l)
                            if p_t not in processor_indicators:
                                processor_indicators[p_t] = {}
                            processor_indicators[p_t][i_name] = v
                            # TODO processor name may be "system::processor"??
                            new_df_rows_idx.append(t2)  # (scenario, system, period, scope, processor, INDICATOR)
                            new_df_rows_data.append((v, None))  # (value, unit)
                    else:
                        for k, v in val.items():
                            l = list(t)
                            l.append(k)
                            t2 = tuple(l)
                            if p_t not in processor_indicators:
                                processor_indicators[p_t] = {}
                            processor_indicators[p_t][k] = v
                            # TODO processor name may be "system::processor"??
                            new_df_rows_idx.append(t2)  # (scenario, system, period, scope, processor, INDICATOR)
                            new_df_rows_data.append((v, None))  # (value, unit)
                else:
                    l = list(t)
                    l.append(indicator.name)
                    t2 = tuple(l)
                    if p_t not in processor_indicators:
                        processor_indicators[p_t] = {}
                    processor_indicators[p_t][indicator.name] = v
                    # TODO processor name may be "system::processor"??
                    new_df_rows_idx.append(t2)  # (scenario, system, period, scope, processor, INDICATOR)
                    new_df_rows_data.append((val, None))  # (value, unit)
        # print(issues)
        # Construct pd.DataFrame with the result of the scalar indicator calculation
        df2 = pd.DataFrame(data=new_df_rows_data,
                           index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=local_idx_names + ["Indicator"]),
                           columns=["Value", "Unit"])
        return df2

    def calculate_global_scalar_indicator(indicator: Indicator) -> pd.DataFrame:
        """
        One global indicator

        :param indicator:
        :return:
        """

        df = results_for_global

        # Parse the expression
        ast = string_to_ast(indicator_expression, indicator.formula if case_sensitive else indicator.formula.lower())

        # Scenario parameters
        scenario_params = create_dictionary()
        for scenario_name, scenario_exp_params in problem_statement.scenarios.items():  # type: str, dict
            scenario_params[scenario_name] = evaluate_parameters_for_scenario(global_parameters, scenario_exp_params)

        issues = []
        new_df_rows_idx = []
        new_df_rows_data = []
        for t, g in df.groupby(global_idx_names):  # GROUP BY Scenario, Period
            params = scenario_params[t[0]]  # Obtain parameter values from scenario, in t[0]
            # TODO Local indicators from the selected processors, for the selected Scenario, Period, Scope
            local_indicators_extract = pd.DataFrame()
            # TODO If a specific indicator or interface from a processor is mentioned, put it as a variable
            # Variables for aggregator functions (which should be present in the AST)
            d = dict(_processors_map=p_map,
                     _processors_dom=serialized_model,
                     _df_group=g,
                     _df_indicators_group=local_indicators_extract)
            # Include parameters (with priority)
            d.update(params)
            if not case_sensitive:
                d = {k.lower(): v for k, v in d.items()}

            state = State(d)
            val, variables = ast_evaluator(ast, state, None, issues, allowed_functions=global_functions_extended)

            if val is not None:
                new_df_rows_idx.append(t)  # (scenario, period)
                new_df_rows_data.append((indicator.name, val, None))
        logging.debug(issues)
        # Construct pd.DataFrame with the result of the scalar indicator calculation
        df2 = pd.DataFrame(data=new_df_rows_data,
                           index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=global_idx_names),
                           columns=["Indicator", "Value", "Unit"])
        return df2

    # -----------------------------------------------------------------------------------------------------------------
    # MAIN of CALCULATE_SCALAR_INDICATORS
    registry, _, _, _, _ = get_case_study_registry_objects(global_state)

    local_idx_to_change = ["Interface"]
    global_idx_to_change = []
    results_for_global = results.copy()
    results.reset_index(local_idx_to_change, inplace=True)
    results_for_global.reset_index(global_idx_to_change, inplace=True)

    # An entry per processor. For each entry, a dictionary of indicators calculated for the processor
    processor_indicators = {}
    # An entry per system. For each system, a dictionary with global indicators (global indicators can be
    # per-system, or all-systems)
    global_indicators = {}
    dfs = []  # List of tuples (dataframe, True if Local else False)
    # For each ScalarIndicator...
    for si in indicators:
        if si._indicator_category == IndicatorCategories.factors_expression:
            try:
                dfi = calculate_local_scalar_indicator(si)
                if not dfi.empty:
                    dfs.append((dfi, True))
            except Exception as e:
                traceback.print_exc()
        elif si._indicator_category == IndicatorCategories.case_study:
            try:
                dfi = calculate_global_scalar_indicator(si)
                if not dfi.empty:
                    dfs.append((dfi, False))
            except Exception as e:
                traceback.print_exc()

    # Restore index
    results.set_index(local_idx_to_change, append=True, inplace=True)

    if len([df[0] for df in dfs if dfs[1]]) > 0:
        local_df = pd.concat([df[0] for df in dfs if dfs[1]])
    else:
        local_df = pd.DataFrame()

    if len([df[0] for df in dfs if not dfs[1]]) > 0:
        # Merge all the results
        global_df = pd.concat([df[0] for df in dfs if not dfs[1]], axis=0, sort=False)
        global_df.set_index(global_idx_names, inplace=True)
    else:
        global_df = pd.DataFrame()

    return local_df, global_df


range_ast = {}


def get_benchmark_category(b: Benchmark, v):
    c = None
    for r in b.ranges.values():
        cat = r["category"]
        range = r["range"]
        if range in range_ast:
            ast = range_ast[range]
        else:
            ast = string_to_ast(number_interval, range)
            range_ast[range] = ast
        in_left = (ast["left"] == "[" and ast["number_left"] <= v) or (ast["left"] == "(" and ast["number_left"] < v)
        in_right = (ast["right"] == "]" and ast["number_right"] >= v) or (
                ast["right"] == ")" and ast["number_right"] > v)
        if in_left and in_right:
            c = cat
            break

    return c


def calculate_local_benchmarks(df_local_indicators, indicators: List[Indicator]):
    """
    From the dataframe of local indicators: scenario, period, scope, processor, indicator, value
    Prepare a dataframe with columns: scenario, period, scope, processor, indicator, benchmark, value

    :param df_local_indicators:
    :param indicators: List of all Indicators (inside it is filtered to process only Local Indicators)
    :return:
    """
    if df_local_indicators.empty:
        return pd.DataFrame()

    ind_map = create_dictionary()
    for si in indicators:
        if si._indicator_category == IndicatorCategories.factors_expression:
            if len(si.benchmarks) > 0:
                ind_map[si.name] = si

    idx_names = ["Scenario", "Period", "Scope", "Processor", "Indicator"]  # Changing factors

    new_df_rows_idx = []
    new_df_rows_data = []
    indicator_column_idx = 4
    value_column_idx = df_local_indicators.columns.get_loc("Value")
    unit_column_idx = df_local_indicators.columns.get_loc("Unit")
    for r in df_local_indicators.itertuples():
        if r[0][indicator_column_idx] in ind_map:
            ind = ind_map[r[0][indicator_column_idx]]
            val = r[1 + value_column_idx]
            unit = r[1 + unit_column_idx]
            for b in ind.benchmarks:
                c = get_benchmark_category(b, val)
                if not c:
                    c = f"<out ({val})>"

                new_df_rows_idx.append(r[0])  # (scenario, period, scope, processor)
                new_df_rows_data.append((val, b.name, c))

    # Construct pd.DataFrame with the result of the scalar indicator calculation
    df2 = pd.DataFrame(data=new_df_rows_data,
                       index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=idx_names),
                       columns=["Value", "Benchmark", "Category"])

    return df2


def calculate_global_benchmarks(df_global_indicators, indicators: List[Indicator]):
    """
    From the dataframe of global indicators: scenario, period, indicator, value
    Prepare a dataframe with columns: scenario, period, indicator, benchmark, value

    :param df_local_indicators:
    :param glb_idx:
    :return:
    """
    if df_global_indicators.empty:
        return pd.DataFrame()

    ind_map = create_dictionary()
    for si in indicators:
        if si._indicator_category == IndicatorCategories.case_study:
            if len(si.benchmarks) > 0:
                ind_map[si.name] = si

    idx_names = ["Scenario", "Period"]  # Changing factors

    new_df_rows_idx = []
    new_df_rows_data = []
    indicator_column_idx = df_global_indicators.columns.get_loc("Indicator")
    value_column_idx = df_global_indicators.columns.get_loc("Value")
    unit_column_idx = df_global_indicators.columns.get_loc("Unit")
    for r in df_global_indicators.itertuples():
        indic = r[1 + indicator_column_idx]
        ind = ind_map[indic]
        val = r[1 + value_column_idx]
        unit = r[1 + unit_column_idx]
        for b in ind.benchmarks:
            c = get_benchmark_category(b, val)
            if not c:
                c = f"<out ({val})>"

            new_df_rows_idx.append(r[0])  # (scenario, period, scope, processor)
            new_df_rows_data.append((indic, val, b.name, c))

    # Construct pd.DataFrame with the result of the scalar indicator calculation
    df2 = pd.DataFrame(data=new_df_rows_data,
                       index=pd.MultiIndex.from_tuples(new_df_rows_idx, names=idx_names),
                       columns=["Indicator", "Value", "Benchmark", "Category"])

    return df2


def prepare_matrix_indicators(indicators: List[MatrixIndicator],
                              registry: PartialRetrievalDictionary,
                              serialized_model: lxml.etree._ElementTree, p_map: Dict,
                              interface_results: pd.DataFrame, indicator_results: pd.DataFrame,
                              dynamic_scenario: bool) -> Dict[str, Dataset]:
    """
    Compute Matrix Indicators

    :param indicators:
    :param registry:
    :param serialized_model:
    :param p_map:
    :param interface_results: The pd.DataFrame with all the interface results
    :param indicator_results: The pd.DataFrame with all the local scalar indicators
    :param dynamic_scenario: True if the matrices have to be prepared for a dynamic scenario
    :return: A dictionary <dataset_name> -> <dataset>
    """

    def prepare_matrix_indicator(indicator: MatrixIndicator) -> pd.DataFrame:
        """
        Compute a Matrix Indicator

        :param indicator: the MatrixIndicator to consider
        :param serialized_model: model as result of parsing XML serialization of model (Processors and Interfaces; no relationships nor observations)
        :param results: result of graph solver
        :return: a pd.DataFrame containing the desired matrix indicator
        """
        # Filter "Scope", if defined
        indicators_df = indicator_results
        if indicator.scope:
            # TODO Consider case sensitiveness of "indicator.scope" (it is entered by the user)
            interfaces_df = interface_results.query('Scope in ("' + indicator.scope + '")')
            if not indicator_results.empty:
                indicators_df = indicator_results.query(f'Scope in ("{indicator.scope}")')
        else:
            interfaces_df = interface_results

        # Apply XPath to obtain the dataframe filtered by the desired set of processors
        dfs, selected_processors = obtain_subset_of_processors(indicator.processors_selector, serialized_model,
                                                               registry, p_map, [interfaces_df, indicators_df])
        interfaces_df, indicators_df = dfs[0], dfs[1]

        # Filter Interfaces
        if indicator.interfaces_selector:
            ifaces = set([_.strip() for _ in indicator.interfaces_selector.split(",")])
            if not case_sensitive:
                ifaces = set([_.lower() for _ in ifaces])

            i_names = get_adapted_case_dataframe_filter(interface_results, "Interface", ifaces)
            # i_names = results.index.unique(level="Interface").values
            # i_names_case = [_ if case_sensitive else _.lower() for _ in i_names]
            # i_names_corr = dict(zip(i_names_case, i_names))
            # i_names = [i_names_corr[_] for _ in ifaces]
            # Filter dataframe to only the desired Interfaces.
            interfaces_df = interfaces_df.query('Interface in [' + ', '.join(['"' + _ + '"' for _ in i_names]) + ']')

        # Filter ScalarIndicators
        if indicator.indicators_selector:
            inds = set([_.strip() for _ in indicator.indicators_selector.split(",")])
            if not case_sensitive:
                inds = set([_.lower() for _ in inds])

            i_names = get_adapted_case_dataframe_filter(indicator_results, "Indicator", inds)
            indicators_df = indicators_df.query('Indicator in [' + ', '.join(['"' + _ + '"' for _ in i_names]) + ']')

        # Filter Attributes
        if indicator.attributes_selector:
            attribs = set([_.strip() for _ in indicator.attributes_selector.split(",")])
            if not case_sensitive:
                attribs = set([_.lower() for _ in attribs])

            # Attributes
            i_names = get_adapted_case_dataframe_filter(interface_results, "Interface", attribs)
            attributes_df = interfaces_df.query('Interface in [' + ', '.join(['"' + _ + '"' for _ in i_names]) + ']')

        # Pivot Table: Dimensions (rows) are (Scenario, Period, Processor[, Scope])
        #              Dimensions (columns) are (Interface, Orientation -of Interface-)
        #              Measures (cells) are (Value)
        idx_columns = ["Scenario", "Period", "Processor"]
        if indicator.scope:
            idx_columns.append("Scope")
        interfaces_df = interfaces_df.pivot_table(values="Value", index=idx_columns,
                                                  columns=["Interface", "Orientation"])
        # Flatten columns, concatenating levels
        interfaces_df.columns = [f"{x} {y}" for x, y in zip(interfaces_df.columns.get_level_values(0),
                                                            interfaces_df.columns.get_level_values(1))]

        if not indicators_df.empty:
            indicators_df = indicators_df.pivot_table(values="Value", index=idx_columns, columns=["Indicator"])
            interfaces_df = pd.merge(interfaces_df, indicators_df, how="outer", left_index=True, right_index=True)

        return interfaces_df

    # For each MatrixIndicator...
    result = {}
    for mi in indicators:
        ds_name = mi.name
        if dynamic_scenario:
            ds_name = "dyn_" + ds_name
        try:
            df = prepare_matrix_indicator(mi)
        except Exception as e:
            df = pd.DataFrame()
            traceback.print_exc()
        if not df.empty:
            result[ds_name] = df
    return result


def get_eum_dataset(dataframe: pd.DataFrame) -> "Dataset":
    """
    Function for a specific case study during MAGIC project.
    Not general purpose, currently it is not used.

    :param dataframe:
    :return:
    """
    # EUM columns
    df = dataframe.query('Orientation == "Input" and '
                         'Interface in ["Biofuel", "CropProduction", "Fertilizer", "HA", "LU"]')

    # EUM rows
    df = df.query(
        'Processor in ['
        '"Society", "Society.Biodiesel", "Society.Bioethanol", '
        '"Society.CommerceImports", "Society.CommerceExports", '
        '"Society.Bioethanol.Cereals", '
        '"Society.Bioethanol.Cereals.Wheat", "Society.Bioethanol.Cereals.Maize", '
        '"Society.Bioethanol.Cereals.ExternalWheat", "Society.Bioethanol.Cereals.ExternalMaize", '
        '"Society.Bioethanol.SugarCrops", '
        '"Society.Bioethanol.SugarCrops.SugarBeet", "Society.Bioethanol.SugarCrops.SugarCane", '
        '"Society.Bioethanol.SugarCrops.ExternalSugarBeet", "Society.Bioethanol.SugarCrops.ExternalSugarCane", '
        '"Society.Biodiesel.OilCrops", '
        '"Society.Biodiesel.OilCrops.PalmOil", "Society.Biodiesel.OilCrops.RapeSeed", '
        '"Society.Biodiesel.OilCrops.SoyBean", '
        '"Society.Biodiesel.OilCrops.ExternalPalmOil", "Society.Biodiesel.OilCrops.ExternalRapeSeed", '
        '"Society.Biodiesel.OilCrops.ExternalSoyBean"'
        ']'
    )

    df = df.pivot_table(values="Value", index=["Scenario", "Period", "Processor", "Level"], columns="Interface")

    # Adding units to column name
    # TODO: remove hardcoded
    df = df.rename(columns={"Biofuel": "Biofuel (tonnes)",
                            "CropProduction": "CropProduction (tonnes)",
                            "Fertilizer": "Fertilizer (kg)",
                            "HA": "HA (h)",
                            "LU": "LU (ha)"})

    logging.debug(df)

    return get_dataset(df, "end_use_matrix", "End use matrix")


def get_dataset(dataframe: pd.DataFrame, code: str, description: str) -> "Dataset":
    ds = Dataset()
    ds.data = dataframe.reset_index()
    ds.code = code
    ds.description = description
    ds.attributes = {}
    ds.metadata = None
    ds.database = None

    if dataframe.index.names[0] != None:
        for dimension in dataframe.index.names:  # type: str
            d = Dimension()
            d.code = dimension
            d.description = None
            d.attributes = None
            d.is_time = (dimension.lower() == "period")
            d.is_measure = False
            cl = dataframe.index.unique(level=dimension).values
            d.code_list = CodeList.construct(
                dimension, dimension, [""],
                codes=[CodeImmutable(c, c, "", []) for c in cl]
            )
            d.dataset = ds

    for measure in dataframe.columns.values:  # type: str
        d = Dimension()
        d.code = measure
        d.description = None
        d.attributes = None
        d.is_time = False
        d.is_measure = True
        d.dataset = ds

    return ds