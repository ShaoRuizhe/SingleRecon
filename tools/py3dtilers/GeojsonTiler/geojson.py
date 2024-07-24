# -*- coding: utf-8 -*-
import PIL
import numpy as np
from ...py3dtilers.earclip import triangulate

from ..Common import Feature, FeatureList


# The GeoJson file contains the ground surface of urban elements, mainly buildings.
# Those elements are called "features", each feature has its own ground coordinates.
# The goal here is to take those coordinates and create a box from it.
# To do this, we compute the center of the lower face
# Then we create the triangles of this face
# and duplicate it with a Z offset to create the upper face
# Then we create the side triangles to connect the upper and the lower faces
class Geojson(Feature):
    """
    The Python representation of a GeoJSON feature.
    A Geojson instance has a geometry and properties.
    """

    n_feature = 0

    # Default height will be used if no height is found when parsing the data
    default_height = 10

    # Default Z will be used if no Z is found in the feature coordinates
    default_z = 0

    # Those values are used to set the color of the features
    attribute_values = list()  # Contains all the values of a semantic attribute
    attribute_min = np.Inf  # Contains the min value of a numeric attribute
    attribute_max = np.NINF  # Contains the max value of a numeric attribute

    def __init__(self, id=None, feature_properties=None, feature_geometry=None):
        super().__init__(id)

        self.feature_properties = feature_properties
        self.feature_geometry = feature_geometry

        self.height = 0
        """How high we extrude the polygon when creating the 3D geometry"""

        self.polygon = list()
        self.custom_triangulation = False

    def custom_triangulate(self, coordinates):
        """
        Custom triangulation method used when we triangulate buffered lines.
        :param coordinates: an array of 3D points ([x, y, Z])

        :return: a list of triangles
        """
        triangles = list()
        length = len(coordinates)

        for i in range(0, (length // 2) - 1):
            triangles.append([coordinates[i], coordinates[length - 1 - i], coordinates[i + 1]])
            triangles.append([coordinates[i + 1], coordinates[length - 1 - i], coordinates[length - 2 - i]])

        return triangles

    def set_z(self, coordinates, z):
        """
        Set the Z value of each coordinate of the feature.
        The Z can be the name of a property to read in the feature or a float.
        :param coordinates: the coordinates of the feature
        :param z: the value of the z
        """
        z_value = Geojson.default_z
        if z != 'NONE':
            if z.replace('.', '', 1).isdigit():
                z_value = float(z)
            else:
                if z in self.feature_properties and self.feature_properties[z] is not None:
                    z_value = self.feature_properties[z]
                elif self.feature_properties[z] is None:
                    z_value = Geojson.default_z
                else:
                    print("No propertie called " + z + " in feature " + str(Geojson.n_feature) + ". Set Z to default value (" + str(Geojson.default_z) + ").")
        for coord in coordinates:
            if len(coord) < 3:
                coord.append(z_value)
            elif z != 'NONE':
                coord[2] = z_value

    def parse_geojson(self, target_properties, is_roof=False, color_attribute=('NONE', 'numeric')):
        """
        Parse a feature to extract the height and the coordinates of the feature.
        :param target_properties: the names of the properties to read
        :param Boolean is_roof: False when the coordinates are on floor level
        """
        # Current feature number (used for debug)
        Geojson.n_feature += 1

        # If precision is equal to 9999, it means Z values of the features are missing, so we skip the feature
        prec_name = target_properties[target_properties.index('prec') + 1]
        if prec_name != 'NONE' and prec_name in self.feature_properties and self.feature_properties[prec_name] is not None:
            if self.feature_properties[prec_name] >= 9999.:
                return False

        height_name = target_properties[target_properties.index('height') + 1]
        if height_name.replace('.', '', 1).isdigit():
            self.height = float(height_name)
        else:
            if height_name in self.feature_properties:
                if self.feature_properties[height_name] is not None:
                    self.height = self.feature_properties[height_name]
                else:
                    self.height = Geojson.default_height
            else:
                print("No propertie called " + height_name + " in feature " + str(Geojson.n_feature) + ". Set height to default value (" + str(Geojson.default_height) + ").")
                self.height = Geojson.default_height

        if color_attribute[0] in self.feature_properties:
            attribute = self.feature_properties[color_attribute[0]]
            if color_attribute[1] == 'numeric':
                if attribute > Geojson.attribute_max:
                    Geojson.attribute_max = attribute
                if attribute < Geojson.attribute_min:
                    Geojson.attribute_min = attribute
            else:
                if attribute not in Geojson.attribute_values:
                    Geojson.attribute_values.append(attribute)

    def parse_geom(self):
        """
        Creates the 3D extrusion of the feature.
        """
        height = self.height
        coordinates = self.polygon
        length = len(coordinates)

        triangles = list()
        vertices = [None] * (2 * length)

        for i, coord in enumerate(coordinates):
            vertices[i] = np.array([coord[0], coord[1], coord[2]])
            vertices[i + length] = np.array([coord[0], coord[1], coord[2] + height])

        # Triangulate the feature footprint
        if self.custom_triangulation:
            poly_triangles = self.custom_triangulate(coordinates)
        else:
            poly_triangles = triangulate(coordinates)

        # Create upper face triangles
        for tri in poly_triangles:
            upper_tri = [np.array([coord[0], coord[1], coord[2] + height]) for coord in tri]
            triangles.append(upper_tri)

        if height>0:
            # Create side triangles
            for i in range(0, length):
                triangles.append([vertices[i], vertices[length + i], vertices[length + ((i + 1) % length)]])
                triangles.append([vertices[i], vertices[length + ((i + 1) % length)], vertices[((i + 1) % length)]])

        self.geom.triangles.append(triangles)
        # self.geom.triangles.append([[np.array([1,0]),np.array([0,0]),np.array([0,1])]]*len(triangles))
        # self.geom.triangles.append([[np.array([0,0.5]),np.array([0,0]),np.array([0.5,0])]]*57+
        #                            [[np.array([1,0.5]),np.array([1,1]),np.array([0.5,1])]]*(len(triangles)-57))
        # self.geom.triangles.append([[np.array([0,0.5]),np.array([0,0]),np.array([0.5,0])]]*len(triangles))
        # self.geom.triangles.append([[np.array([0,1]),np.array([0.,1]),np.array([0.,1])]]*len(triangles))
        # self.geom.triangles.append([[np.array([0.625, 0.5]), np.array([0.375, 0.25]), np.array([0.375, 0.5])]]*len(triangles))
        # uvs=self.get_texture_uvs(triangles,[0,0,1,0.4],[0,0.8,1,1])# 图片的上0.4是顶，下0.2是墙
        # uvs=self.get_texture_uvs(triangles,[0,0,0.333,0.45],[0,0.45,1,1])# 图片的上0.4是顶，下0.2是墙
        uvs = self.get_texture_uvs(triangles, [0., 0., self.feature_properties['roof_texture_area_right'],
                                               self.feature_properties['roof_texture_area_bottom']],
                                   [0, self.feature_properties['roof_texture_area_bottom'],
                                    self.feature_properties['wall_texture_area_right'], 1.])  # 图片的上0.4是顶，下0.2是墙
        self.geom.triangles.append(uvs)

        self.set_box()

    def get_texture_uvs(self,triangles,roof_texture_area=[0.,0.,1.,1.],wall_texture_area=[0.,0.,1.,1.]):
        """
        对一个建筑物形式的三维triangles，获取其对应的texture在图像上的范围。
        其中对三维triangle的要求因为是按照triangle坐标对齐的，因此不一定严格是上面parse_geom形成的形式，只要是有前半部分triangle是顶，后部分triangle是直立墙面（直上直下，或者说roof和footpring完全一致，并且墙面triangle顺序排列）即可
        texture的要求是上方一块是roof，下方一块是墙面，再两个area参数中确定好范围
        Args:
            roof_texture_area: [left,top,right,bottom] 在0,1区间的图片范围
            wall_texture_area: [left,top,right,bottom]
            triangles:三维的trangles
        """
        # todo:按照比例来从图中取出纹理，保证3d的纹理和图片上面的纹理xy比例一致
        roof_triangles = []
        max_x_coor = -float('inf')
        min_x_coor = float('inf')
        max_y_coor = -float('inf')
        min_y_coor = float('inf')
        for i, triangle in enumerate(triangles):
            if (triangle[0][2] != 0 and triangle[1][2] != 0 and triangle[2][2] != 0)\
                    or triangle[0][2] == triangle[1][2] == triangle[2][2] == 0:# 对于要绘制底图的情形
                roof_triangles.append(i)
                for vex in triangle:
                    if vex[0] > max_x_coor:
                        max_x_coor = vex[0]
                    if vex[1] > max_y_coor:
                        max_y_coor = vex[1]
                    if vex[0] < min_x_coor:
                        min_x_coor = vex[0]
                    if vex[1] < min_y_coor:
                        min_y_coor = vex[1]
        uvs = []
        for _ in range(len(triangles)):
            uvs.append([np.array([0., 1.]), np.array([0., 1.]), np.array([0., 1.])])
        roof_x_scale=roof_texture_area[2]-roof_texture_area[0]
        roof_y_scale=roof_texture_area[3]-roof_texture_area[1]
        # 计算roof对应的纹理范围
        for i in roof_triangles:
            for j in range(3):
                uvs[i][j][0] = (triangles[i][j][0] - min_x_coor) / (max_x_coor - min_x_coor) * roof_x_scale + \
                               roof_texture_area[0]
                uvs[i][j][1] = (1-(triangles[i][j][1] - min_y_coor) / (max_y_coor - min_y_coor)) * roof_y_scale + \
                               roof_texture_area[1] # 图片的y坐标是反过来的，注意修正

        if len(triangles)>len(roof_triangles): #要绘制底图时，无需墙面
            # 区分墙面triangle里面位于上面和下面的点，并且记录一周点的位置——step（其实就是上面parse_geom的coordinates，这里在从3d triangle中提取出来）。
            # 记录方式是记录点在多边形一周上面的长度，这也是为了方便对应到后面texture，texture是展开的一圈墙面
            upper_girth = 0
            under_grith = 0
            step = [0]
            indicator = []
            for k in range(len(roof_triangles), len(triangles)):
                upper_points = np.where(np.array([triangles[k][0][2], triangles[k][1][2], triangles[k][2][2]]) != 0)[0]
                if len(upper_points) == 2:
                    upper_girth += np.linalg.norm(triangles[k][upper_points[0]][:2] - triangles[k][upper_points[1]][:2])
                    step.append((upper_girth))
                    indicator.append(1)
                else:
                    under_points = np.delete([0, 1, 2], upper_points[0])
                    under_grith += np.linalg.norm(triangles[k][under_points[0]][:2] - triangles[k][under_points[1]][:2])
                    indicator.append(0)

            assert indicator[0] == 1
            assert upper_girth == under_grith

            # 计算墙面triangle对应texture的图片坐标
            step =1- np.array(step) / upper_girth #环绕的方向似乎是顺时针（从上到下看），因此x方向的也要反向一下
            wall_x_scale=wall_texture_area[2]-wall_texture_area[0]
            for k in range(len(triangles) - len(roof_triangles)):
                if k % 2 == 0:  # upper
                    uvs[k + len(roof_triangles)][0] = np.array(
                        [step[k // 2] * wall_x_scale + wall_texture_area[0], wall_texture_area[3]])# 同样要注意映射将y左边反过来
                    uvs[k + len(roof_triangles)][1] = np.array(
                        [step[k // 2] * wall_x_scale + wall_texture_area[0], wall_texture_area[1]])
                    uvs[k + len(roof_triangles)][2] = np.array(
                        [step[k // 2 + 1] * wall_x_scale + wall_texture_area[0], wall_texture_area[1]])
                else:  # under
                    uvs[k + len(roof_triangles)][0] = np.array(
                        [step[k // 2] * wall_x_scale + wall_texture_area[0], wall_texture_area[3]])
                    uvs[k + len(roof_triangles)][1] = np.array(
                        [step[k // 2 + 1] * wall_x_scale + wall_texture_area[0], wall_texture_area[1]])
                    uvs[k + len(roof_triangles)][2] = np.array(
                        [step[k // 2 + 1] * wall_x_scale + wall_texture_area[0], wall_texture_area[3]])
        return uvs

    def get_geojson_id(self):
        return super().get_id()

    def set_geojson_id(self, id):
        return super().set_id(id)


class Geojsons(FeatureList):
    """
    A decorated list of Geojson instances.
    """

    def __init__(self, objects=None):
        super().__init__(objects)

    @staticmethod
    def parse_geojsons(features, properties, is_roof=False, color_attribute=('NONE', 'numeric')):
        """
        Create 3D features from the GeoJson features.
        :param features: the features to parse from the GeoJSON
        :param properties: the properties used when parsing the features
        :param is_roof: substract the height from the features coordinates

        :return: a list of triangulated Geojson instances.
        """
        feature_list = list()

        for feature in features:
            if not feature.parse_geojson(properties, is_roof, color_attribute):
                continue

            # Create geometry as expected from GLTF from an geojson file
            feature.parse_geom()
            feature_list.append(feature)

            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\R-C.png')
            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\R-C.jpg')
            # texture = PIL.Image.open(r'C:\Users\srz\Desktop\b-w.png')
            # texture = PIL.Image.open(r'C:\Users\srz\Desktop\yrb.png')
            # texture = PIL.Image.open('../../tests/obj_tiler_data/TexturedCube/tex.png')
            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\w.png')
            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\b.png')
            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\split.png')
            # texture=PIL.Image.open(r'C:\Users\srz\Desktop\texture.png')

            texture=PIL.Image.open(feature.feature_properties['texture_img'])
            feature.set_texture(texture)

        return Geojsons(features)
        # return Geojsons(features[0:1])
