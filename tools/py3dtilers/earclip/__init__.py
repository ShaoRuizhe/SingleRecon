#Code copied from mrbaozi.
#https://github.com/mrbaozi/triangulation and turned into a module.
        
def IsConvex(a, b, c):
    # only convex if traversing anti-clockwise!
    crossp = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if crossp >= 0:
        return True 
    return False 

def InTriangle(a, b, c, p):
    L = [0, 0, 0]
    eps = 0.0000001
    # calculate barycentric coefficients for point p
    # eps is needed as error correction since for very small distances denom->0
    L[0] = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) \
          /(((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])) + eps)
    L[1] = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) \
          /(((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])) + eps)
    L[2] = 1 - L[0] - L[1]
    # check if p lies in triangle (a, b, c)
    for x in L:
        if x >= 1 or x <= 0:
            return False  
    return True  

def IsClockwise(poly):
    # initialize sum with last element
    sum = (poly[0][0] - poly[len(poly)-1][0]) * (poly[0][1] + poly[len(poly)-1][1])
    # iterate over all other elements (0 to n-1)
    for i in range(len(poly)-1):
        sum += (poly[i+1][0] - poly[i][0]) * (poly[i+1][1] + poly[i][1])
    if sum > 0:
        return True
    return False

def GetEar(poly):
    size = len(poly)
    if size < 3:
        return []
    if size == 3:
        tri = (poly[0], poly[1], poly[2])
        del poly[:]
        return tri
    for i in range(size):
        tritest = False
        p1 = poly[(i-1) % size]
        p2 = poly[i % size]
        p3 = poly[(i+1) % size]
        if IsConvex(p1, p2, p3):
            for x in poly:
                if not (x in (p1, p2, p3)) and InTriangle(p1, p2, p3, x):
                    tritest = True
            if tritest == False:
                del poly[i % size]
                return (p1, p2, p3)
    print('GetEar(): no ear found')
    return []

def triangulate(pts):
    """
    Split a convex, simple polygon into triangles, using earclipping algorithm.
    Code literally copied from mrbaozi.
    https://github.com/mrbaozi/triangulation and turned into a module.
    
    Parameters:
        pts: a list of lists (of coordinates). E.g.
          [[ 229.23,   78.21],
           [ 258.49,   17.23],
           [ 132.09,  -22.43],
           [ 107.97,   23.23],
           [  65.85,   28.58],
           [  41.63,  -92.48],
           [   4.68,  204.75],
           [ 176.52,  193.84],
           [ 171.13,   87.15]]
           (note the last point isn't the first, it is assumed closed)
    Returns:
        A list of tuples. Each tuple contains three coordinates (representing
        one triangle), each coordinate is a list of two values. E.g.
            [([171.12, 87.39],
              [176.17, 193.49],
              [4.69,   204.10]),
             ([171.10, 87.22],
              [4.32,   204.87],
              [-12.23, 137.54])]          
            
    """
    tri = []
    plist = pts[::-1] if IsClockwise(pts) else pts[:]
    while len(plist) >= 3:
        a = GetEar(plist)
        if a == []:
            break
        tri.append(a)
    return tri
