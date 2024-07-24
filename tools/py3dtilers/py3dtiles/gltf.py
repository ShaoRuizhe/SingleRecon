# -*- coding: utf-8 -*-
import struct
import numpy as np
import json
from .gltf_material import GlTFMaterial


class GlTF(object):
    HEADER_LENGTH = 12
    CHUNK_HEADER_LENGTH = 8

    def __init__(self):
        self.header = {}
        self.body = None
        self.batch_length = 0

    def to_array(self):  # glb
        scene = json.dumps(self.header, separators=(',', ':'))

        # body must be 4-byte aligned
        scene += ' ' * ((4 - len(scene) % 4) % 4)

        padding = np.array([0 for i in range(0, (4 - len(self.body) % 4) % 4)],
                           dtype=np.uint8)

        length = GlTF.HEADER_LENGTH + (2 * GlTF.CHUNK_HEADER_LENGTH)
        length += len(self.body) + len(scene) + len(padding)
        binaryHeader = np.array([0x46546C67,  # "glTF" magic
                                 2,  # version
                                 length], dtype=np.uint32)
        jsonChunkHeader = np.array([len(scene),  # JSON chunck length
                                    0x4E4F534A], dtype=np.uint32)  # "JSON"

        binChunkHeader = np.array([len(self.body) + len(padding),
                                   # BIN chunck length
                                   0x004E4942], dtype=np.uint32)  # "BIN"

        return np.concatenate((binaryHeader.view(np.uint8),
                               jsonChunkHeader.view(np.uint8),
                               np.frombuffer(scene.encode('utf-8'), dtype=np.uint8),
                               binChunkHeader.view(np.uint8),
                               self.body,
                               padding))

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        glTF : GlTf
        """

        glTF = GlTF()

        if struct.unpack("4s", array[0:4])[0] != b"glTF":
            raise RuntimeError("Array does not contain a binary glTF")

        version = struct.unpack("i", array[4:8])[0]
        if version != 1 and version != 2:
            raise RuntimeError("Unsupported glTF version")

        length = struct.unpack("i", array[8:12])[0]
        json_chunk_length = struct.unpack("i", array[12:16])[0]

        chunk_type = struct.unpack("i", array[16:20])[0]
        if chunk_type != 0 and chunk_type != 1313821514:  # 1313821514 => 'JSON'
            raise RuntimeError("Unsupported binary glTF content type")

        index = GlTF.HEADER_LENGTH + GlTF.CHUNK_HEADER_LENGTH  # Skip the header and the JSON chunk header
        header = struct.unpack(str(json_chunk_length) + "s",
                               array[index:index + json_chunk_length])[0]
        glTF.header = json.loads(header.decode("ascii"))

        index += json_chunk_length + GlTF.CHUNK_HEADER_LENGTH  # Skip the JSON chunk data and the binary chunk header
        glTF.body = array[index:length]

        return glTF

    @staticmethod
    def from_binary_arrays(arrays, transform, binary=True, batched=True,
                           uri=None, materials=[GlTFMaterial()]):
        """
        Parameters
        ----------
        arrays : array of dictionaries
            Each dictionary has the data for one geometry
            arrays['position']: binary array of vertex positions
            arrays['normal']: binary array of vertex normals
            arrays['uv']: binary array of vertex texture coordinates
                          (Not implemented yet)
            arrays['bbox']: geometry bounding box (numpy.array)
            arrays['matIndex']: the index of the material used by the geometry
            arrays['vertex_color']: the vertex colors

        transform : numpy.array
            World coordinates transformation flattend matrix

        Returns
        -------
        glTF : GlTF
        """

        glTF = GlTF()
        nMaterials = len(materials)
        textured = 'uv' in arrays[0]
        vertex_colored = 'vertex_color' in arrays[0]
        binVertices = [[] for _ in range(nMaterials)]
        binNormals = [[] for _ in range(nMaterials)]
        binIds = [[] for _ in range(nMaterials)]
        binUvs = [[] for _ in range(nMaterials)]
        binColors = [[] for _ in range(nMaterials)]
        nVertices = [0 for _ in range(nMaterials)]
        bb = [[] for _ in range(nMaterials)]
        glTF.batch_length = 0
        for i, geometry in enumerate(arrays):
            matIndex = geometry['matIndex']
            binVertices[matIndex].append(geometry['position'])
            binNormals[matIndex].append(geometry['normal'])
            n = round(len(geometry['position']) / 12)
            nVertices[matIndex] += n
            bb[matIndex].append(geometry['bbox'])
            if batched:
                binIds[matIndex].append(np.full(n, i, dtype=np.float32))
            if textured:
                binUvs[matIndex].append(geometry['uv'])
            if vertex_colored:
                binColors[matIndex].append(geometry['vertex_color'])

        if batched:
            glTF.batch_length = len(arrays)
            for i in range(0, len(binVertices)):
                binVertices[i] = b''.join(binVertices[i])
                binNormals[i] = b''.join(binNormals[i])
                binUvs[i] = b''.join(binUvs[i])
                binIds[i] = b''.join(binIds[i])
                binColors[i] = b''.join(binColors[i])
                [minx, miny, minz] = bb[i][0][0]
                [maxx, maxy, maxz] = bb[i][0][1]
                for box in bb[i][1:]:
                    minx = min(minx, box[0][0])
                    miny = min(miny, box[0][1])
                    minz = min(minz, box[0][2])
                    maxx = max(maxx, box[1][0])
                    maxy = max(maxy, box[1][1])
                    maxz = max(maxz, box[1][2])
                bb[i] = [[minx, miny, minz], [maxx, maxy, maxz]]

        glTF.header = compute_header(binVertices, nVertices, bb, transform,
                                     textured, batched, glTF.batch_length, uri, materials, vertex_colored)
        glTF.body = np.frombuffer(compute_binary(binVertices, binNormals,
                                  binIds, binUvs, binColors), dtype=np.uint8)

        return glTF


def compute_binary(binVertices, binNormals, binIds, binUvs, binColors):
    bv = b''.join(binVertices)
    bn = b''.join(binNormals)
    bid = b''.join(binIds)
    buv = b''.join(binUvs)
    bc = b''.join(binColors)
    return bv + bn + buv + bid + bc


def compute_header(binVertices, nVertices, bb, transform,
                   textured, batched, batchLength, uri, meshMaterials, vertex_colored):
    # Buffer
    meshNb = len(binVertices)
    sizeVce = []
    for i in range(0, meshNb):
        sizeVce.append(len(binVertices[i]))

    byteLength = 2 * sum(sizeVce)
    if textured:
        byteLength += int(round(2 * sum(sizeVce) / 3))
    if batched:
        byteLength += int(round(sum(sizeVce) / 3))
    buffers = [{
        'byteLength': byteLength
    }]
    if uri is not None:
        buffers["binary_glTF"]["uri"] = uri

    # Buffer view
    bufferViews = []
    # vertices
    bufferViews.append({
        'buffer': 0,
        'byteLength': sum(sizeVce),
        'byteOffset': 0,
        'target': 34962
    })
    bufferViews.append({
        'buffer': 0,
        'byteLength': sum(sizeVce),
        'byteOffset': sum(sizeVce),
        'target': 34962
    })
    if textured:
        bufferViews.append({
            'buffer': 0,
            'byteLength': int(round(2 * sum(sizeVce) / 3)),
            'byteOffset': 2 * sum(sizeVce),
            'target': 34962
        })
    if batched:
        bufferViews.append({
            'buffer': 0,
            'byteLength': int(round(sum(sizeVce) / 3)),
            'byteOffset': (2 * sum(sizeVce)) + (int(textured) * int(round(2 * sum(sizeVce) / 3))),
            'target': 34962
        })
    if vertex_colored:
        bufferViews.append({
            'buffer': 0,
            'byteLength': sum(sizeVce),
            'byteOffset': (2 * sum(sizeVce)) + (int(textured) * int(round(2 * sum(sizeVce) / 3))) + (int(batched) * int(round(sum(sizeVce) / 3))),
            'target': 34962
        })

    # Accessor
    accessors = []
    for i in range(0, meshNb):
        # vertices
        accessors.append({
            'bufferView': 0,
            'byteOffset': sum(sizeVce[0:i]),
            'componentType': 5126,
            'count': nVertices[i],
            'min': [bb[i][0][0], bb[i][0][1], bb[i][0][2]],
            'max': [bb[i][1][0], bb[i][1][1], bb[i][1][2]],
            'type': "VEC3"
        })
        # normals
        accessors.append({
            'bufferView': 1,
            'byteOffset': sum(sizeVce[0:i]),
            'componentType': 5126,
            'count': nVertices[i],
            'max': [1, 1, 1],
            'min': [-1, -1, -1],
            'type': "VEC3"
        })
        if textured:
            accessors.append({
                'bufferView': 2,
                'byteOffset': int(round(2 / 3 * sum(sizeVce[0:i]))),
                'componentType': 5126,
                'count': sum(nVertices),
                'max': [1, 1],
                'min': [0, 0],
                'type': "VEC2"
            })
        if batched:
            accessors.append({
                'bufferView': 2 + int(textured),
                'byteOffset': int(round(1 / 3 * sum(sizeVce[0:i]))),
                'componentType': 5126,
                'count': nVertices[i],
                'max': [batchLength],
                'min': [0],
                'type': "SCALAR"
            })
        if vertex_colored:
            accessors.append({
                'bufferView': 2 + int(textured) + int(batched),
                'byteOffset': sum(sizeVce[0:i]),
                'componentType': 5126,
                'count': nVertices[i],
                'max': [1, 1, 1],
                'min': [0, 0, 0],
                'type': "VEC3"
            })

    # Meshes
    meshes = []
    nAttributes = 2 + int(textured) + int(batched) + int(vertex_colored)
    for i in range(0, meshNb):
        meshes.append({
            'primitives': [{
                'attributes': {
                    "POSITION": nAttributes * i,
                    "NORMAL": (nAttributes * i) + 1
                },
                "material": i,
                "mode": 4
            }]
        })
        if textured:
            meshes[i]['primitives'][0]['attributes']['TEXCOORD_0'] = (nAttributes * i) + 2
        if batched:
            meshes[i]['primitives'][0]['attributes']['_BATCHID'] = (nAttributes * i) + 2 + int(textured)
        if vertex_colored:
            meshes[i]['primitives'][0]['attributes']['COLOR_0'] = (nAttributes * i) + 2 + int(textured) + int(batched)

    # Nodes
    nodes = []
    for i in range(0, meshNb):
        nodes.append({
            'matrix': [float(e) for e in transform],
            'mesh': i
        })

    # Materials
    images = []
    materials = []
    for i, mat in enumerate(meshMaterials):
        material = mat.to_dict('Material' + str(i), len(images))
        if mat.is_textured():
            images.append({'uri': mat.textureUri})
        materials.append(material)

    # Final glTF
    header = {
        'asset': {
            "generator": "py3dtiles",
            "version": "2.0"
        },
        'scene': 0,
        'scenes': [{
            'nodes': [i for i in range(0, len(nodes))]
        }],
        'nodes': nodes,
        'meshes': meshes,
        'materials': materials,
        'accessors': accessors,
        'bufferViews': bufferViews,
        'buffers': buffers
    }

    # Texture data
    if textured:
        header['textures'] = [{
            'sampler': 0,
            'source': 0
        }]
        header['images'] = images
        header['samplers'] = [{
            "magFilter": 9729,
            "minFilter": 9987,
            "wrapS": 10497,
            "wrapT": 10497
        }]

    return header
