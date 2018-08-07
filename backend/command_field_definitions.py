import re

from backend import CommandField
from backend.command_generators.basic_elements_parser import simple_ident, unquoted_string, alphanums_string, \
    hierarchy_expression_v2, key_value_list, key_value, parameter_value, parameter_domain, expression_with_parameters, \
    time_expression, boolean, indicator_expression

data_types = ["Number", "Boolean", "URL", "UUID", "Datetime", "String", "UnitName", "Category", "Geo"]
parameter_types = ["Number", "Category", "Boolean"]
element_types = ["Parameter", "Processor", "InterfaceType", "Interface"]
spheres = ["Biosphere", "Technosphere"]
roegen_types = ["Flow", "Fund"]
orientations = ["Input", "Output"]
functional_or_structural = ["Functional", "Structural"]
instance_or_archetype = ["Instance", "Archetype"]
relation_types = [# Relationships between Processors
                  "is-a",  # "Left" gets a copy of ALL "Right" interface types
                  "as-a",  # Left must already have ALL interfaces from Right. Similar to "part-of" in the sense that ALL Right interfaces are connected from Left to Right
                  "part-of", "|",  # The same thing. Left is inside Right. No assumptions on flows between child and parent.
                  "aggregate", "compose",
                  "association",
                  # Relationships between interfaces
                  ">", "<"
                  ]
instantiation_types = ["Upscale"]

commands = {
    "CatHierarchies":
    [CommandField(allowed_names=["Source"], name="source", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["HierarchyGroup"], name="hierarchy_group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["HierarchyName"], name="hierarchy_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Level", "LevelCode"], name="level", mandatory=False, allowed_values=None, parser=alphanums_string),
     CommandField(allowed_names=["ReferredHierarchy"], name="referred_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Code"], name="code", mandatory=False, allowed_values=None, parser=alphanums_string),
     # NOTE: Removed because parent code must be members of the same hierarchy in definition
     # CommandField(allowed_names=["ReferredHierarchyParent"], name="referred_hierarchy_parent", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ParentCode"], name="parent_code", mandatory=False, allowed_values=None, parser=alphanums_string),
     CommandField(allowed_names=["Label"], name="label", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Expression", "Formula"], name="expression", mandatory=False, allowed_values=None, parser=hierarchy_expression_v2),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute", "<attr_name>"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    "CatHierarchiesMapping":
    [CommandField(allowed_names=["SourceDataset"], name="source_dataset", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceHierarchy"], name="source_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceCode"], name="source_code", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationHierarchy"], name="destination_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationCode"], name="destination_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Weight"], name="weight", mandatory=True, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Context"], name="context", mandatory=True, allowed_values=None, parser=context_query),  # "context_query" allows applying different weights depending on values of dimensions of the source dataset
     ],
    "AttributeTypes":
    [CommandField(allowed_names=["AttributeTypeName"], name="attribute_type_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DataType"], name="data_type", mandatory=True, allowed_values=data_types, parser=simple_ident),
     CommandField(allowed_names=["ElementTypes"], name="element_types", mandatory=False, allowed_values=element_types, parser=key_value),
     CommandField(allowed_names=["Domain"], name="domain", mandatory=False, allowed_values=None, parser=domain_definition)  # "domain_definition" for Category and NUmber. Boolean is only True or False. Other data types cannot be easily constrained (URL, UUID, Datetime, Geo, String)
     ],
    "DatasetDef":
    [CommandField(allowed_names=["DatasetName"], name="dataset_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DatasetDataLocation"], name="dataset_data_location", mandatory=True, allowed_values=None, parser=url),
     CommandField(allowed_names=["DatasetDataDescription"], name="dataset_description", mandatory=True, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["ConceptType"], name="concept_type", mandatory=True, allowed_values=["Dimension", "Measure", "Attribute"], parser=simple_ident),
     CommandField(allowed_names=["ConceptName"], name="concept_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ConceptDataType"], name="concept_data_type", mandatory=True, allowed_values=data_types, parser=simple_ident),
     CommandField(allowed_names=["ConceptDomain"], name="concept_domain", mandatory=False, allowed_values=None, parser=domain_definition),
     CommandField(allowed_names=["ConceptDescription"], name="concept_description", mandatory=True, allowed_values=None, parser=unquoted_string),
    ],
    # "DatasetData" needs a specialized parser
    "AttributeSets":
    [CommandField(allowed_names=["AttributeSetName"], name="attribute_set_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute", "<attr_name>"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    "Parameters":
    [CommandField(allowed_names=["Name", "ParameterName"], name="name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Type"], name="type", mandatory=True, allowed_values=parameter_types, parser=simple_ident),
     CommandField(allowed_names=["Domain"], name="domain", mandatory=False, allowed_values=None, parser=parameter_domain),
     CommandField(allowed_names=["Value"], name="value", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute", "<attr_name>"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    # "DatasetQry" needs a specialized parser
    "InterfaceTypes":
    [CommandField(allowed_names=["InterfaceTypeHierarchyName"], name="interface_type_hierarchy_name", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InterfaceTypeName"], name="interface_type_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Sphere"], name="sphere", mandatory=True, allowed_values=spheres, parser=simple_ident),
     CommandField(allowed_names=["RoegenType"], name="roegen_type", mandatory=True, allowed_values=roegen_types, parser=simple_ident),
     CommandField(allowed_names=["Orientation"], name="orientation", mandatory=False, allowed_values=orientations, parser=simple_ident),
     CommandField(allowed_names=["ParentInterfaceType"], name="parent_interface_type", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Formula", "Expression"], name="formula", mandatory=False, allowed_values=None, parser=hierarchy_expression_v2),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Unit"], name="unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute", "<attr_name>"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    "Processors":
    [CommandField(allowed_names=["ProcessorName"], name="processor_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ProcessorGroup"], name="processors_group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["FunctionalOrStructural"], name="functional_or_structural", mandatory=False, allowed_values=functional_or_structural, parser=simple_ident),
     CommandField(allowed_names=["InstanceOrArchetype"], name="processors_group", mandatory=False, allowed_values=instance_or_archetype, parser=simple_ident),
     CommandField(allowed_names=["ParentProcessorName"], name="parent_processor_name", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Location"], name="location", mandatory=False, allowed_values=None, parser=geo_value),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute", "<attr_name>"], name="attribute", mandatory=False, allowed_values=None, parser=key_value),
     ],
    "Interfaces":
    [CommandField(allowed_names=["SpecificName"], name="specific_name", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InterfaceType"], name="interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Processor"], name="processor", mandatory=True, allowed_values=None, parser=processor_ident),
     CommandField(allowed_names=["Sphere"], name="sphere", mandatory=False, allowed_values=spheres, parser=simple_ident),
     CommandField(allowed_names=["RoegenType"], name="roegen_type", mandatory=False, allowed_values=roegen_types, parser=simple_ident),
     CommandField(allowed_names=["Orientation"], name="orientation", mandatory=False, allowed_values=orientations, parser=simple_ident),
     CommandField(allowed_names=["InterfaceAttributes"], name="interface_attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["InterfaceAttribute", "<attr_name>"], name="interface_attribute", mandatory=False, allowed_values=None, parser=key_value),
     CommandField(allowed_names=["Value"], name="value", mandatory=False, allowed_values=None, parser=None),
     CommandField(allowed_names=["Unit"], name="unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["Uncertainty"], name="uncertainty", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Assessment"], name="assessment", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["PedigreeMatrix"], name="pedigree_matrix", mandatory=False, allowed_values=None, parser=reference_name),
     CommandField(allowed_names=["Pedigree"], name="pedigree", mandatory=False, allowed_values=None, parser=pedigree_code),
     CommandField(allowed_names=["RelativeTo"], name="relative_to", mandatory=False, allowed_values=None, parser=simple_ident_plus_unit_name),
     CommandField(allowed_names=["Time"], name="time", mandatory=False, allowed_values=None, parser=time_expression),
     CommandField(allowed_names=["ProcessorLocation"], name="processor_location", mandatory=False, allowed_values=None, parser=geo_value),
     CommandField(allowed_names=["Source"], name="qq_source", mandatory=False, allowed_values=None, parser=reference_name_or_unquoted_string),
     CommandField(allowed_names=["NumberAttributes"], name="number_attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["NumberAttribute", "<attr_name>"], name="number_attribute", mandatory=False, allowed_values=None, parser=key_value),
     CommandField(allowed_names=["Comments"], name="comments", mandatory=False, allowed_values=None, parser=unquoted_string),
     ],
    "Relationships":
    [CommandField(allowed_names=["SourceProcessor"], name="source_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceInterface"], name="source_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["TargetProcessor"], name="target_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["TargetInterface"], name="target_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["RelationType"], name="relation_type", mandatory=False, allowed_values=relation_types, parser=simple_ident),
     CommandField(allowed_names=["Weight"], name="flow_weight", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["SourceCardinality"], name="source_cardinality", mandatory=False, allowed_values=None, parser=cardinality),
     CommandField(allowed_names=["TargetCardinality"], name="target_cardinality", mandatory=False, allowed_values=None, parser=cardinality)
     ],
    "Instantiations":
    [CommandField(allowed_names=["InvokingProcessor"], name="invoking_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["RequestedProcessor"], name="requested_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InstantiationType"], name="instantiation_type", mandatory=True, allowed_values=instantiation_types, parser=simple_ident),
     CommandField(allowed_names=["UpscaleParentInterface"], name="upscale_parent_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["UpscaleChildInterface"], name="upscale_child_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["UpscaleWeight"], name="upscale_weight", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["UpscaleParentContext"], name="upscale_parent_context", mandatory=False, allowed_values=None, parser=upscale_context),
     CommandField(allowed_names=["UpscaleChildContext"], name="upscale_child_context", mandatory=False, allowed_values=None, parser=upscale_context),
     ],
    "ScaleChangers":
    [CommandField(allowed_names=["SourceHierarchy"], name="source_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceInterfaceType"], name="source_interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["TargetHierarchy"], name="target_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["TargetInterfaceType"], name="target_interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InvokingProcessor"], name="invoking_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceContext"], name="source_context", mandatory=False, allowed_values=None, parser=scale_context),
     CommandField(allowed_names=["TargetContext"], name="target_context", mandatory=False, allowed_values=None, parser=scale_context),
     CommandField(allowed_names=["Scale"], name="scale", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["SourceUnit"], name="source_unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["TargetUnit"], name="target_unit", mandatory=False, allowed_values=None, parser=unit_name),
     ],
    "SharedElements":
    [

     ],
    "ReusedElements":
    [

     ],
    "Indicators":
    [CommandField(allowed_names=["IndicatorName"], name="indicator_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["MultipleProcessors"], name="multiple_processors", mandatory=True, allowed_values=None, parser=boolean),
     CommandField(allowed_names=["Formula", "Expression"], name="expression", mandatory=True, allowed_values=None, parser=indicator_expression),
     CommandField(allowed_names=["Benchmark"], name="benchmark", mandatory=False, allowed_values=None, parser=benchmark_definition),
     ],
    # "ProblemStatement" needs a specialized parser
}


def compile_command_field_regexes():
    for cmd in commands:
        # Compile the regular expressions of column names
        flags = re.IGNORECASE
        for c in commands[cmd]:
            rep = [re.escape(r) for r in c.allowed_names]
            c.regex_allowed_names = re.compile("|".join(rep), flags=flags)


compile_command_field_regexes()