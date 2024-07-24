import math
import os.path

import PIL.Image
import cv2
import numpy as np
import shapely
import shapely.affinity
import skimage.morphology
from pycocotools.coco import COCO
from PIL import Image, ImageDraw


def compute_distances(distance_maps):
    distance_maps.sort(axis=2)
    distance_maps = distance_maps[:, :, :2]
    distances = np.sum(distance_maps, axis=2)
    return distances

def draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)

def draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    assert type(polygons) == list, "polygons should be a list"
    assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygons = [shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0)) for polygon in polygons]
        line_width *= 2
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(polygon.exterior.coords, fill=255)
            for interior in polygon.interiors:
                draw.polygon(interior.coords, fill=0)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(polygon.exterior.coords, fill=255, width=line_width)
            for interior in polygon.interiors:
                draw.line(interior.coords, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in polygon.exterior.coords:
                draw_circle(draw, vertex, line_width / 2, fill=255)
            for interior in polygon.interiors:
                for vertex in interior.coords:
                    draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array

def draw_polygons_offndir(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    assert type(polygons) == list, "polygons should be a list"
    assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygons = [shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0)) for polygon in polygons]
        line_width *= 2
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(polygon.exterior.coords, fill=255)
            for interior in polygon.interiors:
                draw.polygon(interior.coords, fill=0)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(polygon.exterior.coords, fill=255, width=line_width)
            for interior in polygon.interiors:
                draw.line(interior.coords, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in polygon.exterior.coords:
                draw_circle(draw, vertex, line_width / 2, fill=255)
            for interior in polygon.interiors:
                for vertex in interior.coords:
                    draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array

def compute_raster_distances_sizes(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    """
    Returns:
         - distances: sum of distance to closest and second-closest annotation for each pixel.
         - size_weights: relative size (normalized by image area) of annotation the pixel belongs to.
    """
    assert type(polygons) == list, "polygons should be a list"

    # Filter out zero-area polygons
    polygons = [polygon for polygon in polygons if 0 < polygon.area]

    # tic = time.time()

    channel_count = fill + edges + vertices
    polygons_raster = np.zeros((*shape, channel_count), dtype=np.uint8)
    distance_maps = np.ones((*shape, len(polygons)))  # Init with max value (distances are normed)
    sizes = np.ones(shape)  # Init with max value (sizes are normed)
    image_area = shape[0] * shape[1]
    for i, polygon in enumerate(polygons):
        minx, miny, maxx, maxy = polygon.bounds
        mini = max(0, math.floor(miny) - 2*line_width)
        minj = max(0, math.floor(minx) - 2*line_width)
        maxi = min(polygons_raster.shape[0], math.ceil(maxy) + 2*line_width)
        maxj = min(polygons_raster.shape[1], math.ceil(maxx) + 2*line_width)
        bbox_shape = (maxi - mini, maxj - minj)
        bbox_polygon = shapely.affinity.translate(polygon, xoff=-minj, yoff=-mini)
        bbox_raster = draw_polygons([bbox_polygon], bbox_shape, fill, edges, vertices, line_width, antialiasing)
        polygons_raster[mini:maxi, minj:maxj] = np.maximum(polygons_raster[mini:maxi, minj:maxj], bbox_raster)
        bbox_mask = 0 < np.sum(bbox_raster, axis=2)  # Polygon interior + edge + vertex
        if bbox_mask.max():  # Make sure mask is not empty
            polygon_mask = np.zeros(shape, dtype=np.bool)
            polygon_mask[mini:maxi, minj:maxj] = bbox_mask
            polygon_dist = cv2.distanceTransform(1 - polygon_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5,
                                        dstType=cv2.CV_64F)
            polygon_dist /= (polygon_mask.shape[0] + polygon_mask.shape[1])  # Normalize dist
            distance_maps[:, :, i] = polygon_dist

            selem = skimage.morphology.disk(line_width)
            bbox_dilated_mask = skimage.morphology.binary_dilation(bbox_mask, selem=selem)
            sizes[mini:maxi, minj:maxj][bbox_dilated_mask] = polygon.area / image_area

    polygons_raster = np.clip(polygons_raster, 0, 255)
    # skimage.io.imsave("polygons_raster.png", polygons_raster)

    if edges:
        edge_channels = -1 + fill + edges
        # Remove border edges because they correspond to cut buildings:
        polygons_raster[:line_width, :, edge_channels] = 0
        polygons_raster[-line_width:, :, edge_channels] = 0
        polygons_raster[:, :line_width, edge_channels] = 0
        polygons_raster[:, -line_width:, edge_channels] = 0

    distances = compute_distances(distance_maps)
    # skimage.io.imsave("distances.png", distances)

    distances = distances.astype(np.float16)
    sizes = sizes.astype(np.float16)

    # toc = time.time()
    # print(f"Rasterize {len(polygons)} polygons: {toc - tic}s")

    return polygons_raster, distances, sizes

def init_angle_field(polygons, shape, line_width=1):
    """
    Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
    Angle between 0 and pi.
    This is not invariant to symmetries.

    :param polygons:
    :param shape:
    :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
              mask: np.array((num_edge_pixels, 2), dtype=np.int))
    """
    assert type(polygons) == list, "polygons should be a list"
    if len(polygons):
        assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for polygon in polygons:
        draw_linear_ring(draw, polygon.exterior, line_width)
        for interior in polygon.interiors:
            draw_linear_ring(draw, interior, line_width)

    # Convert image to numpy array
    array = np.array(im)
    return array

def draw_linear_ring(draw, linear_ring, line_width):
    # --- edges:
    coords = np.array(linear_ring.xy).transpose()
    edge_vect_array = np.diff(coords, axis=0)
    edge_angle_array = np.angle(edge_vect_array[:, 1] + 1j * edge_vect_array[:, 0])  # ij coord sys
    neg_indices = np.where(edge_angle_array < 0)
    edge_angle_array[neg_indices] += np.pi

    first_uint8_angle = None
    for i in range(coords.shape[0] - 1):
        edge = (coords[i], coords[i + 1])
        angle = edge_angle_array[i]
        uint8_angle = int((255 * angle / np.pi).round())
        if first_uint8_angle is None:
            first_uint8_angle = uint8_angle
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)

    # Add first vertex back on top (equals to last vertex too):
    draw_circle(draw, line[1], radius=line_width / 2, fill=first_uint8_angle)

# read Bonai
bonai_data_root = r'D:\Documents\Dataset\building_footprint\BONAI/'
json_file=os.path.join(bonai_data_root,'coco/bonai_beijing_trainval.json')
img_path=os.path.join(bonai_data_root,'trainval\images')
coco_data=COCO(json_file)
image_info=coco_data.loadAnns()
# PFFL make up

for i,img_anns in coco_data.imgToAnns.items():
    img_shape=(coco_data.imgs[i]['height'],coco_data.imgs[i]['width'])
    img = PIL.Image.open(os.path.join(img_path,coco_data.imgs[i]['file_name']))
    polygons=[]
    for img_ann in img_anns:
        polygons.append(shapely.geometry.Polygon(np.array(img_ann['segmentation']).reshape(-1,2)))
    out=compute_raster_distances_sizes(polygons,img_shape,line_width=10)
    out_angle=init_angle_field(polygons,img_shape,4)

    pass

# save to file
pass