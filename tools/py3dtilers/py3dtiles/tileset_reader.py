import json
import sys
import os

from .utils import TileReader
from .bounding_volume_box import BoundingVolumeBox
from .tile import Tile
from .tileset import TileSet


class TilesetReader(object):

    def __init__(self):
        self.tile_reader = TileReader()

    def read_tilesets(self, paths):
        """
        Read tilesets.
        :param paths: the paths to the tilesets
        :return: a list of TileSet
        """
        tilesets = list()
        for path in paths:
            try:
                tilesets.append(self.read_tileset(path))
            except Exception:
                print("Couldn't read the tileset", path)
        return tilesets

    def read_tileset(self, path):
        """
        param: path: Path to a directory containing the tileset.json
        """
        with open(os.path.join(path, 'tileset.json')) as f:
            json_tileset = json.load(f)
        root = json_tileset['root']
        tileset = TileSet()

        for child in root['children']:
            self.read_tile(child, tileset, path, 0)

        tileset.get_root_tile().set_bounding_volume(BoundingVolumeBox())
        return tileset

    def read_tile(self, json_tile, parent, path, depth=0):
        tile = Tile()
        tile.set_geometric_error(json_tile['geometricError'])
        uri = os.path.join(path, json_tile['content']['uri'])
        tile.set_content(self.tile_reader.read_file(uri))
        tile.set_transform(json_tile['transform'])
        tile.set_refine_mode(json_tile['refine'])

        if 'box' in json_tile['boundingVolume']:
            bounding_volume = BoundingVolumeBox()
            bounding_volume.set_from_list(json_tile['boundingVolume']['box'])
        else:
            print('Sphere and region bounding volumes aren\'t supported')
            sys.exit(1)
        tile.set_bounding_volume(bounding_volume)

        if depth == 0:
            parent.add_tile(tile)
        else:
            parent.add_child(tile)

        if 'children' in json_tile:
            for child in json_tile['children']:
                self.read_tile(child, tile, path, depth + 1)
