# -*- coding: utf-8 -*-
import sys
import os
import pathlib


class SchemaWithSample:
    """
    This is the gathering of a Json schema (file) together with a sample
    that this schema should validate. The sole usage of an instance of
    this class is to be used with the method
          SchemaValidators::register_schema_with_sample()
    """

    def __init__(self, key):
        """
        :param key: the name of the class (implementing ThreeDTilesNotion)
               to be used as access key to retrieve the associated validator.
        """
        self.key = key
        self.schema_directory = None
        self.schema_file_name = None
        self.schema_file_path = None
        self.sample = None

    def get_key(self):
        return self.key

    def assert_schema_file_path(self):
        if os.path.isfile(self.schema_file_path):
            return
        print(f'Unfound schema file {self.schema_file_name}')
        sys.exit(1)

    def sync_schema_file_path(self):
        if not self.schema_directory:
            print('A directory must be set prior to syncing file_path.')
            sys.exit(1)
        self.schema_file_path = os.path.join(self.schema_directory,
                                             self.schema_file_name)
        self.assert_schema_file_path()

    def set_schema_file_path(self, schema_file_path):
        """
        :param schema_file_path: path to the file holding the schema that (at
               some point) will have to be registered together to the list
               of known schemas (look for a line of the form
                  "Draft7Validator(schema, resolver = self.resolver")
               WARNING: for the time being there is a strong constrain placed
               on the schema_file_path that MUST be relative to the python
               package.
               Warning: when the json schema held in that file uses references
               (that is entries of the form $ref) to other external schemas,
               then those (json) schemas must be encountered in the
               SAME directory as the schema itself (otherwise the schema
               reference resolver has no clue on where to find the sub-schemas)
        """
        self.schema_file_path = schema_file_path
        self.assert_schema_file_path()

    def get_schema_file_path(self):
        return self.schema_file_path

    def set_directory(self, directory):
        # Refer to the code documentation of SchemaValidators.__init__() for
        # the comment on pre-install/post-install concerns.

        if os.path.isdir(directory):
            self.schema_directory = directory
            return
        # We had no success in local. Could this be an installed package ?
        installed_path = pathlib.Path(__file__).parent.absolute()
        self.schema_directory = os.path.join(installed_path, "..", directory)
        if not os.path.isdir(self.schema_directory):
            print('Failed to establish an installed package context:')
            print(f'unfound directory pathes {directory} and {self.schema_directory}.')
            print('Exiting.')
            sys.exit(1)

    def get_directory(self):
        return self.schema_directory

    def set_filename(self, filename):
        self.schema_file_name = filename
        self.sync_schema_file_path()

    def get_filename(self):
        return self.schema_file_name

    def set_sample(self, sample):
        self.sample = sample

    def get_sample(self):
        return self.sample
