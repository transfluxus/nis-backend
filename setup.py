# ONE TIME  -----------------------------------------------------------------------------
#
# pip install --upgrade setuptools wheel twine
#
# Create account:
# PyPI test: https://test.pypi.org/account/register/
# or PyPI  : https://pypi.org/account/register/
#
# EACH TIME -----------------------------------------------------------------------------
#
# Modify version code in "setup.py" (this file)
#
# Build (cd to directory where "setup.py" is)
# python3 setup.py sdist bdist_wheel
#
# Upload:
# PyPI test: twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*
# or PyPI  : twine upload --skip-existing dist/*
#
# INSTALL   ------------------------------------------------------------------------------
#
# PyPI test: pip install --index-url https://test.pypi.org/simple/ --upgrade nexinfosys-backend
# PyPI     : pip install --upgrade nexinfosys-backend
# No PyPI  : pip install -e <local path where "setup.py" (this file) is located>
#
# EXECUTE (example. "gunicorn" must be installed: "pip install gunicorn")
# (IT WORKS WITH ONLY 1 WORKER!!!)
# gunicorn --workers=1 --log-level=debug --timeout=2000 --bind 0.0.0.0:8081 nexinfosys.restful_service.service_main:app
#

from pkg_resources import yield_lines
from setuptools import setup
package_name = 'nexinfosys-backend'
version = '0.33'


def parse_requirements(strs):
    """Yield ``Requirement`` objects for each specification in `strs`

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(strs))

    ret = []
    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if ' #' in line:
            line = line[:line.find(' #')]
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith('\\'):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        ret.append(line)

    return ret


with open('requirements.txt') as f:
    required = f.read().splitlines()

install_reqs = parse_requirements(required)
print(install_reqs)

setup(
    name=package_name,
    version=version,
    install_requires=install_reqs,
    packages=['nexinfosys', 'nexinfosys.common', 'nexinfosys.models', 'nexinfosys.models.experiments', 'nexinfosys.solving',
              'nexinfosys.solving.graph', 'nexinfosys.ie_exports', 'nexinfosys.ie_imports', 'nexinfosys.ie_imports.data_sources',
              'nexinfosys.ie_imports.experimental', 'nexinfosys.authentication', 'nexinfosys.model_services',
              'nexinfosys.restful_service', 'nexinfosys.restful_service.gunicorn', 'nexinfosys.restful_service.mod_wsgi',
              'nexinfosys.command_executors', 'nexinfosys.command_executors.misc', 'nexinfosys.command_executors.solving',
              'nexinfosys.command_executors.analysis', 'nexinfosys.command_executors.version2',
              'nexinfosys.command_executors.read_query', 'nexinfosys.command_executors.external_data',
              'nexinfosys.command_executors.specification', 'nexinfosys.command_generators',
              'nexinfosys.command_generators.spreadsheet_command_parsers',
              'nexinfosys.command_generators.spreadsheet_command_parsers.analysis',
              'nexinfosys.command_generators.spreadsheet_command_parsers.external_data',
              'nexinfosys.command_generators.spreadsheet_command_parsers.specification',
              'nexinfosys.command_generators.spreadsheet_command_parsers_v2', 'nexinfosys.magic_specific_integrations'],
    include_package_data=True,
    url='https://github.com/MAGIC-nexus/nis-backend',
    license='BSD-3',
    author='rnebot',
    author_email='rnebot@itccanarias.org',
    description='A package supporting MuSIASEM formalism and methodology'
)