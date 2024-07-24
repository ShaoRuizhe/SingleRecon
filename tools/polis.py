import sys
import time
from copy import copy
from collections import Counter

import click
import fiona
from shapely import geometry
from rtree import index
import numpy as np
            
def compare_polys(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.

    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.

    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.

    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis(bndry_a.coords, bndry_b)
    dist += polis(bndry_b.coords, bndry_a)
    return dist

def compare_multi_polys(multi_poly_a, multi_poly_b):
    """Compares multi polygons via the "polis" distance metric.

    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.

    Input:
        poly_a: A Shapely MultiPolygon.
        poly_b: Another Shapely MultiPolygon.

    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = multi_poly_a.boundary, multi_poly_b.boundary
    dist = multi_polis(bndry_a, bndry_b)
    dist += multi_polis(bndry_b, bndry_a)
    return dist

def multi_polis(multi_bndry_a,multi_bndry_b):
    sum=0.0
    num_potins=0
    for single_bndry in multi_bndry_a:
        num_potins+=len(single_bndry.coords)
        for pt in (geometry.Point(c) for c in single_bndry.coords[:-1]):
            sum+= multi_bndry_b.distance(pt)
    return sum/float(2*num_potins)


def polis(coords, bndry):
    """Computes one side of the "polis" metric.

    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).
    
    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (geometry.Point(c) for c in coords[:-1]): # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum/float(2*len(coords))


def shp_to_list(shpfile):
    """Dumps all the geometries from the shapefile into a list.

    This makes it quick to build the spatial index!
    """
    with fiona.open(shpfile) as src:
        return [geometry.shape(rec['geometry']) for rec in src]

@click.command()
@click.argument('in_ref', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('in_cmp', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('out', type=click.Path(dir_okay=False, writable=True, resolve_path=True))
def score(in_ref, in_cmp, out):
    """Given two polygon vector files, calculate the polis score
    between them. The third argument specifies the output file, which
    contains the same geometries as in_cmp, but with the polis score
    assigned to the geometry.

    We consider the first vector file to be the reference.

    Example (from the root of the git repo):

    $polis data/FullSubset.shp data/user_data.shp out.shp
    """
    # Read in all the geometries in the reference shapefile.
    ref_polys = shp_to_list(in_ref)

    # Build a spatial index of the reference shapes.
    idx = index.Index((i, geom.bounds, None) for i, geom in enumerate(ref_polys))
    
    # Input data to measure.
    hits = []
    with fiona.open(in_cmp) as src:
        meta = copy(src.meta)
        meta['schema']['properties'] = {'polis': 'float:15.2'}
        with fiona.open(out, 'w', **src.meta) as sink:
            for rec in src:
                cmp_poly = geometry.shape(rec['geometry'])
                ref_pindices = [i for i in idx.nearest(cmp_poly.bounds)]

                # Limit how many we check, if given an excessive
                # number of ties (aka if someone put in a huge
                # bounding box that covered a ton of geometries.)
                if len(ref_pindices) > 5: 
                    ref_pindices = ref_pindices[:5]
                scores = [compare_polys(cmp_poly, ref_polys[i]) for i in ref_pindices]

                polis_score = min(scores)
                hits.append(ref_pindices[scores.index(polis_score)])

                sink.write({'geometry':rec['geometry'],
                            'properties':{'polis': polis_score}})

    # Summarize results.
    print("Number of matches: {}".format(len(hits)))
    print("Number of misses: {}".format(len(ref_polys) - len(hits)))
    print("Duplicate matches: {}".format(sum([1 for i in Counter(hits).values() if i > 1])))

def polis_metric(cmp_polygons,ref_polygons):
    # 对应好的多边形一个对一个计算polis
    assert len(cmp_polygons)==len(ref_polygons)
    time0=time.time()
    print('totole polygon number:%d,evaluating polis_metric...'%len(cmp_polygons))
    scores = [compare_polys(cmp_polygons[i], ref_polygons[i]) for i in range(len(cmp_polygons))]
    # import matplotlib.pyplot as plt
    # for i,score in enumerate(scores):
    #     plt.plot(*cmp_polygons[i].exterior.xy,color='red')
    #     plt.plot(*ref_polygons[i].exterior.xy,color='blue')
    #     plt.title('polis:%.4f'%score)
    #     plt.show()
    mean_score=np.mean(scores)
    print('finish. time:%.3f. polis score:%.3f'%(time.time()-time0,mean_score))
    return mean_score

def multi_polygon_polis_metric(cmp_multi_polygons,ref_multi_polygons):
    # 将一张图的多个多边形整合成一个multipolygon对象，和标签的multipolygon对象计算polis
    # time0=time.time()
    # print('totole polygon number:%d,%d,evaluating polis_metric...'%(len(cmp_multi_polygons),len(ref_multi_polygons)))
    score= compare_multi_polys(cmp_multi_polygons,ref_multi_polygons)
    # print('finish. time:%.3f. polis score:%.3f'%(time.time()-time0,score))
    return score

if __name__ == "__main__":
    score()

