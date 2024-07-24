# -*- coding: utf-8 -*-
import numpy
from .threedtiles_notion import ThreeDTilesNotion


class FeatureTable(ThreeDTilesNotion):
    """
    Only the JSON header has been implemented for now. According to the feature
    table documentation, the binary body is useful for storing long arrays of
    data (better performances).
    TODO: Implement a body and a header with their own to_array/from_array
    """

    def __init__(self):
        super().__init__()

    def to_array(self):
        """
        :return: the notion encoded as an array of binaries
        """
        # First encode the concerned attributes as a json string
        as_json = self.to_json()
        # and make sure it respects a mandatory 8-byte alignement (refer e.g.
        # to feature table documentation)
        as_json += ' ' * (8 - (len(as_json) + 28) % 8)
        # eventually return an array of binaries representing the
        # considered ThreeDTilesNotion
        return numpy.frombuffer(as_json.encode(), dtype=numpy.uint8)
