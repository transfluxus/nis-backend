import json
from collections import namedtuple
from typing import List
from sqlalchemy import Column, Integer, String, Unicode, Boolean, ForeignKey
from sqlalchemy import orm, event
from sqlalchemy.orm import relationship, backref

from backend.common.helper import create_dictionary
from backend.common.namedlist import namedlist
from backend.models.musiasem_methodology_support import ORMBase

# Ad-hoc structures
ConceptImmutable = namedtuple("ConceptTuple", "name istime description attributes code_list")
CodeImmutable = namedtuple("CodeTuple", "code description level children")
CodeMutable = namedlist("Code", ["code", "description", "level", "children", "parents"])


class CodeListInMemory:
    """
    Stores a code list
    """
    def __init__(self):
        self._name = None
        self._description = None
        self._levels = None
        self._codes = None

    @staticmethod
    def construct_from_dict(d):
        """
        Dictioanry with keys: "name", "description", "levels", "codes"
        "codes" contains the first level codes, which can successively contain
        :param d:
        :return:
        """
        return CodeListInMemory.construct(d["name"], d["description"], d["levels"], [CodeImmutable(**i) for i in d["codes"].values])

    @staticmethod
    def construct(name: str, description: str, levels: List[str], codes: List[CodeImmutable]):
        """

        :param name: Name of the Code List
        :param description: Description of the Code List
        :param levels: Names of the levels
        :param codes: List of codes, including in each the following tuple: CodeImmutable = namedtuple("CodeTuple", "code description level children")
        :return:
        """
        cl = CodeListInMemory()
        cl._name = name
        cl._description = description
        cl._levels = {l: {} for l in levels}  # Level name: list of codes in the level
        cl._codes = {}  # Flat dictionary of CodeMutable by code, with children and parents
        children = {}
        for i in codes:
            cl._levels[i.level].append(i)
            for c in i.children:
                children[c] = i.code  # Child "c" mentioned by parent "i["code"]"
                if c not in cl._codes:
                    cl._codes[i.code] = CodeMutable(code=i.code, parents=[])
                code = cl._codes[i.code]
                # Add a parent to code
                code.parents.append(i.code)
            if i["code"] not in cl._codes:
                cl._codes[i.code] = CodeMutable(code=i.code, parents=[])
            code = cl._codes[i.code]
            code.children = i.children
            code.level = i.level
            code.description = i.description

        # Check all children are in cl._codes.
        s1 = set([c for c in children])
        s2 = set([c for c in cl._codes])
        sdiff = s1 - s2
        if len(sdiff) > 0:
            s = ["Code '"+children[c]+"' mentions '"+c+"' as child, but '"+c+"' was not detailed" for c in sdiff]
            raise Exception("; ".join(s))

        return cl


class CodeList(ORMBase):
    __tablename__ = "dc_cl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))

    def __init__(self):
        # NOTE: If needed, use CodeListInMemory.construct to initialize .code_list
        self.code_list = None  # type: CodeListInMemory

    @orm.reconstructor
    def init_on_load(self):
        self.code_list = None

    def get_codes(self):
        return self.to_dict().items()

    def to_dict(self):
        d = {}
        for l in self.levels:
            for c in l.codes:
                d[c.code] = c.description
        return d

    @staticmethod
    def construct_from_dict(d):
        return CodeList.construct(d["name"], d["description"], d["levels"], [CodeImmutable(**i) for i in d["codes"].values])

    @staticmethod
    def construct(name: str, description: str, levels: List[str], codes: List[CodeImmutable]):
        """

        :param name: Name of the Code List
        :param description: Description of the Code List
        :param levels: Names of the levels
        :param codes: List of codes, including in each the following tuple: CodeImmutable = namedtuple("CodeTuple", "code description level children")
        :return:
        """

        cl = CodeList()
        cl.code = name
        cl.description = description
        # Levels
        levels_dict = create_dictionary()
        for l in levels:
            cll = CodeListLevel()
            cll.code_list = cl  # Point to the containing CodeList
            cll.code = l
            cll.description = None
            levels_dict[l] = cll
        # Codes
        codes_dict = create_dictionary()
        for ct in codes:
            c = Code()
            c.code = ct.code
            c.description = ct.description
            if ct.level in levels_dict:
                c.level = levels_dict[ct.level]  # Point to the containing CodeListLevel
            else:
                c.level = None
            codes_dict[ct.code] = c
            c.children = []
            c.parents = []
        # Set children & parents
        for ct in codes:
            for ch in ct.children:
                if ch in codes_dict:
                    c.children.append(codes_dict[ch])
                    codes_dict[ch].parents.append(c)

        return cl


class CodeListLevel(ORMBase):
    __tablename__ = "dc_cl_levels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))

    code_list_id = Column(Integer, ForeignKey(CodeList.id))
    code_list = relationship(CodeList, backref=backref("levels", cascade="all, delete-orphan"))


class Code(ORMBase): # A single value and its description for a Dimension
    __tablename__ = "dc_cl_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))

    level_id = Column(Integer, ForeignKey(CodeListLevel.id))
    level = relationship(CodeListLevel, backref=backref("codes", cascade="all, delete-orphan"))

    geography = Column(Unicode)  # If it is a geographical code, what is the boundary
    geography_url = Column(String(1024))

    children = Column(Unicode)  # postgresql.JSONB)  # List of codes totalling this code
    parents = Column(Unicode)  # postgresql.JSONB)  # List of codes aggregating this code


@event.listens_for(Code, 'before_insert')
def code_before_insert(mapper, connection, target):
    target.children = json.dumps(target.children) if target.children else None
    target.parents = json.dumps(target.parents) if target.parents else None


@event.listens_for(Code, "load")
def code_after_load(target, context):
    target.parents = json.loads(target.parents) if target.parents else {}
    target.children = json.loads(target.children) if target.children else {}


class Concept(ORMBase):  # Concepts are independent of datasets
    __tablename__ = "dc_concepts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))
    is_time = Column(Boolean, default=False)  # A concept having time nature

    attributes = Column(Unicode) # postgresql.JSONB)

    parent_id = Column(Integer, ForeignKey("dc_concepts.id"))
    parent = relationship("Concept", backref=backref("children", remote_side=[id]), cascade="all, delete-orphan")

    code_list_emb = Column(Unicode) # postgresql.JSONB)

    # Zero or one Code List
    code_list_id = Column(Integer, ForeignKey(CodeList.id))
    code_list = relationship(CodeList, backref=backref("concept", cascade="all, delete-orphan"))


@event.listens_for(Concept, 'before_insert')
def concept_before_insert(mapper, connection, target):
    target.code_list_emb = json.dumps(target.code_list_emb) if target.code_list_emb else None
    target.attributes = json.dumps(target.attributes) if target.attributes else None


@event.listens_for(Concept, "load")
def concept_after_load(target, context):
    target.attributes = json.loads(target.attributes) if target.attributes else {}
    target.code_list_emb = json.loads(target.code_list_emb) if target.code_list_emb else {}


# --------------------------------------------

class DataSource(ORMBase):  # Eurostat, OECD, FAO, ...
    __tablename__ = "dc_data_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32))
    description = Column(String(1024))
    data_dictionary = Column(Unicode) # postgresql.JSONB)  # "Metadata"


@event.listens_for(DataSource, 'before_insert')
def datasource_before_insert(mapper, connection, target):
    target.data_dictionary = json.dumps(target.data_dictionary) if target.data_dictionary else None


@event.listens_for(DataSource, "load")
def datasource_after_load(target, context):
    target.data_dictionary = json.loads(target.data_dictionary) if target.data_dictionary else {}


class Database(ORMBase): # A data source can have one or more databases
    __tablename__ = "dc_databases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))
    data_dictionary = Column(Unicode) # postgresql.JSONB)  # "Metadata"

    data_source_id = Column(Integer, ForeignKey(DataSource.id))
    data_source = relationship(DataSource)


@event.listens_for(Database, 'before_insert')
def database_before_insert(mapper, connection, target):
    target.data_dictionary = json.dumps(target.data_dictionary) if target.data_dictionary else None


@event.listens_for(Database, "load")
def database_after_load(target, context):
    target.data_dictionary = json.loads(target.data_dictionary) if target.data_dictionary else {}


class Dataset(ORMBase): # A database has many datasets
    __tablename__ = "dc_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(128))
    description = Column(String(1024))
    data_dictionary = Column(Unicode)  # postgresql.JSONB)  # "Metadata"

    attributes = Column(Unicode)  # postgresql.JSONB)

    database_id = Column(Integer, ForeignKey(Database.id))
    database = relationship(Database)

    # TODO Dimensions, Measures
    # TODO Stores

    def __init__(self):
        self.data = None  # type: pd.DataFrame

    def get_objects_list(self):
        """
        A list of objects related to the dataset is elaborated.
        This list is used for serialization/deserialization.
        :return:
        """
        lst = [self]
        if self.database:
            lst.append(self.database)
            if self.database.data_source:
                lst.append(self.database.data_source)
        # TODO Append "Concept" related to each dimension? (a Concept may be related
        for d in self.dimensions:
            lst.append(d)
            if d.code_list:
                lst.append(d.code_list)
                for l in d.code_list.levels:
                    lst.append(l)
                    for c in l.codes:
                        lst.append(c)
        return lst

    @orm.reconstructor
    def init_on_load(self):
        self.data = None


@event.listens_for(Dataset, 'before_insert')
def dataset_before_insert(mapper, connection, target):
    target.data_dictionary = json.dumps(target.data_dictionary) if target.data_dictionary else None
    target.attributes = json.dumps(target.attributes) if target.attributes else None


@event.listens_for(Dataset, "load")
def dataset_after_load(target, context):
    target.attributes = json.loads(target.attributes) if target.attributes else {}
    target.data_dictionary = json.loads(target.data_dictionary) if target.data_dictionary else {}


class Dimension(ORMBase):  # A dimension is a concept linked to a dataset. It can be a Measure (a Measure is a kind of Dimension)
    __tablename__ = "dc_ds_dimensions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32))
    description = Column(String(1024))
    is_time = Column(Boolean, default=False)
    is_measure = Column(Boolean, default=False)
    attributes = Column(Unicode) # postgresql.JSONB)

    dataset_id = Column(Integer, ForeignKey(Dataset.id))
    dataset = relationship(Dataset, lazy='subquery', backref=backref("dimensions", cascade="all, delete-orphan"))

    # Zero or one Concept
    concept_id = Column(Integer, ForeignKey(Concept.id))  # The concept will contain the full code list
    concept = relationship(Concept)

    code_list_emb = Column(Unicode) # postgresql.JSONB)  # A reduced code list of the concept, if it exists
    # Zero or one Code List, this contains the reduced Code List
    code_list_id = Column(Integer, ForeignKey(CodeList.id))
    code_list = relationship(CodeList, backref=backref("dimension", cascade="all, delete-orphan"))


@event.listens_for(Dimension, 'before_insert')
def dimension_before_insert(mapper, connection, target):
    target.code_list_emb = json.dumps(target.code_list_emb) if target.code_list_emb else None
    target.attributes = json.dumps(target.attributes) if target.attributes else None


@event.listens_for(Dimension, "load")
def dimension_after_load(target, context):
    target.attributes = json.loads(target.attributes) if target.attributes else {}
    target.code_list_emb = json.loads(target.code_list_emb) if target.code_list_emb else {}


class Store(ORMBase): # Location for one or more datasets. Datasets point to Stores, also a dataset can be in multiple locations
    __tablename__ = "dc_ds_stores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameters = Column(Unicode) # postgresql.JSONB)

    dataset_id = Column(Integer, ForeignKey(Dataset.id))
    dataset = relationship(Dataset)


@event.listens_for(Store, 'before_insert')
def store_before_insert(mapper, connection, target):
    target.parameters = json.dumps(target.parameters) if target.parameters else None


@event.listens_for(Store, "load")
def store_after_load(target, context):
    target.parameters = json.loads(target.parameters) if target.parameters else {}


"""
Consider also caching and refresh policy
Consider dimensions defined in MuSIASEM, and mappings
"""
