# HSPF Model Plot Routines
#
# David J. Lampert, PhD, PE
#
# Last updated: 04/16/2013
#
# Purpose: Lots of routines here to generate images for development of an HSPF
# model. Descriptions below.
#

import numpy as np, os, pickle, math, csv, datetime

from matplotlib              import pyplot, path,patches,colors, cm, gridspec
from matplotlib.ticker       import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy                   import stats, log10
from itertools               import chain
from shapefile               import Reader, Writer
from PIL                     import Image, ImageDraw

from pyhspf.preprocessing.merge_shapes       import format_shape
from pyhspf.preprocessing.combine_catchments import get_distance
from pyhspf.preprocessing.raster             import get_pixel, get_raster
from pyhspf.preprocessing.raster             import get_raster_in_poly
from pyhspf.preprocessing.raster             import get_raster_on_poly

def is_number(s):
    try: float(s) 
    except ValueError: return False
    return True

def get_aggregate_map(aggregatefile):
    """Creates a dictionary that maps landuse codes to HSPF codes."""

    with open(aggregatefile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader][1:]

    columns = [column for column in zip(*rows)]

    # set up the groups for the unique values in the aggregate file, a 
    # dictionary to map to the HSPF ID types, and the cumulative areas

    codes      = [int(code) for code in columns[0]]
    groups     = [int(code) for code in columns[2]]
    categories = [category for category in columns[3]]

    m         = {code: group for code, group in zip(codes, groups)}
    landtypes = {group: category for group, category in zip(groups, categories)}
    groups = list(set(groups))
    groups.sort()

    return m, landtypes, groups

def shape_to_mask(shape, width = 1000):

    # separate the x and y values for the shape

    xs, ys = zip(*shape)

    x0 = min(xs)
    y0 = min(ys)

    # calculate the pixel height

    height = math.ceil((max(ys) - min(ys)) / (max(xs) - min(xs)) * width)

    # calculate the width and height of a pixel

    w = (max(xs) - min(xs)) / width
    h = (max(ys) - min(ys)) / height

    # convert points of the polygon to pixels

    pixel_polygon = [(get_pixel(x, x0, w), get_pixel(y, y0, h))
                     for x, y in zip(xs, ys)]

    # make a PIL image with the appropriate dimensions to use as a mask 

    rasterpoly = Image.new('L', (width, height), 1)
    rasterize  = ImageDraw.Draw(rasterpoly)

    # rasterize the polygon

    rasterize.polygon(pixel_polygon, 0)

    # convert the PIL array to numpy boolean to use as a mask

    mask = 1 - np.array(rasterpoly)

    return mask

def poly_to_cdf(poly, n = 1000, dim = 'x'):
    """Determines the cumulative distribution function of the area of the 
    polygon (assumed to be made up of a list of points) in the chosen dimension
    for "n" values."""

    # convert the points to xs and ys

    xs, ys = zip(*poly)

    if dim == 'x':

        # convert the points to a mask array of ones (in) and zeros (out)
        
        mask = shape_to_mask(poly, width = n)

        # get the total number of pixels in the shape

        tot = mask.sum()

        # iterate from left to right or top to bottom and get the fraction
        # inside as a function of x or y

        x0 = min(xs)
        delx = (max(xs) - min(xs)) / n
        xrange = [x0 + delx * i for i in range(n + 1)]
        fractions = [column.sum() / tot for column in mask.transpose()]

        cdf = [0]
        for f in fractions: cdf.append(cdf[-1] + f)

        return xrange, cdf

    elif dim == 'y':
        mask = shape_to_mask(poly, width = 1)

    else:
        print('unknown coordinate dimension, please specify x or y.')
        return

def get_boundaries(shapes, space = 0.1):
    """Gets the boundaries for the plot."""

    boundaries = shapes[0].bbox
    for shape in shapes[0:]:
        b = shape.bbox
        if b[0] < boundaries[0]: boundaries[0] = b[0]
        if b[1] < boundaries[1]: boundaries[1] = b[1]
        if b[2] > boundaries[2]: boundaries[2] = b[2]
        if b[3] > boundaries[3]: boundaries[3] = b[3]

    xmin = boundaries[0] - (boundaries[2] - boundaries[0]) * space
    ymin = boundaries[1] - (boundaries[3] - boundaries[1]) * space
    xmax = boundaries[2] + (boundaries[2] - boundaries[0]) * space
    ymax = boundaries[3] + (boundaries[3] - boundaries[1]) * space

    return xmin, ymin, xmax, ymax

def aggregate_hourly_daily(hourly):
    """Aggregates an hourly timeseries to a daily one."""

    if len(hourly) < 24:
        print('insufficiently short time series')
        return

    daily = []
    for i in range(0, len(hourly) - 23, 24):
        daily.append(sum(hourly[i:i+24]))

    return daily

def make_patch(points, facecolor, edgecolor = 'Black', width = 1, alpha = None,
               hatch = None, label = None):
    """Uses a list or array of points to generate a matplotlib patch."""

    vertices = [(point[0], point[1]) for point in points]
    vertices.append((points[0][0], points[0][1]))

    codes     = [path.Path.LINETO for i in range(len(points) + 1)]
    codes[0]  = path.Path.MOVETO

    patch = patches.PathPatch(path.Path(vertices, codes), facecolor = facecolor,
                              edgecolor = edgecolor, lw = width, hatch = hatch,
                              alpha = alpha, label = label)
    return patch

def add_raster(fig, filename, resolution, extent, colormap, intensity, 
               scale = 1):
    """adds a rectangular raster image with corners located at the extents
    to a plot.
    """

    xmin, ymin, xmax, ymax = extent

    xs = np.array([xmin + (xmax - xmin) / resolution * i 
                   for i in range(resolution + 1)])
    ys = np.array([ymax  - (ymax  - ymin)  / resolution * i 
                   for i in range(resolution + 1)])

    zs = np.zeros((resolution + 1, resolution + 1))

    for i in range(len(ys)):
        zs[i, :] = get_raster(filename, zip(xs, [ys[i]] * (resolution + 1)),
                              quiet = True)

    if intensity is not None:
        indices = zs > intensity[0]
        if zs[indices].min() > intensity[1]:
            zs = zs / scale
        norm = colors.Normalize(vmin = intensity[0], vmax = intensity[1])
    elif scale is not None:
        zs = zs / scale
        space = 0.2
        min, max = zs.min(), zs.max()
        min, max = min - space * (max - min), max + space * (max - min)
        norm = colors.Normalize(vmin = min, vmax = max)
    else: norm = None

    # plot the grid

    im = fig.imshow(zs, extent = [xmin, xmax, ymin, ymax], norm = norm, 
                    cmap = colormap)

    return im

def plot_subbasin(directory, HUC8, subbasin, raster = None, catchments = True,
                  flowlines = True, outlets = True, patchcolor = None, 
                  output = None, verbose = True):
    """Makes a plot of all the flowlines and catchments of a basin on top of a
    raster image file."""

    sfile      = directory + '/%s/%d/combined'          % (HUC8, subbasin)
    cfile      = directory + '/%s/%d/catchments'        % (HUC8, subbasin)
    flowfile   = directory + '/%s/%d/flowlines'         % (HUC8, subbasin)
    combined   = directory + '/%s/%d/combined_flowline' % (HUC8, subbasin)
    flowVAAs   = directory + '/%s/flowlineVAAs'         %  HUC8
    outletfile = directory + '/%s/%ssubbasin_outlets'   % (HUC8, HUC8)
    inletfile  = directory + '/%s/%ssubbasin_inlets'    % (HUC8, HUC8)
    dem        = directory + '/%s/%selevations.tif'     % (HUC8, HUC8)
    crop       = directory + '/%s/%scropland.tif'       % (HUC8, HUC8)

    if output is None: 
        output = directory + '/%s/%d/elevations' % (HUC8, subbasin)

    if raster == 'elevation':
        raster_file = dem
        resolution = 200
        colormap = 'gist_earth'
        intensity = [100, 600]

    if raster == 'cropland':
        raster_file = crop
        colormap = 'summer_r'

    if verbose: print('generating plot of subbasin %d\n' % subbasin)

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 11)
    
    # add the title

    if catchments and flowlines: description = 'Catchments and Flowlines'
    elif catchments:             description = 'Catchments'
    elif flowlines:              description = 'Flowlines'
    else:                        description = 'Combined Shapes'

    if raster == 'elevation': description = description + ' on 30 meter DEM'
    if raster == 'cropland':  description = description + ', Crop Land Use'

    title = ('Cataloging Unit %s, Subbasin %d\n%s' % 
             (HUC8, subbasin, description))
    subplot.set_title(title, size = 14)

    # open up and show the catchments and the subbasin boundary

    if patchcolor is None: facecolor = (1, 0, 0, 0.)
    else:                  facecolor = patchcolor

    s = Reader(sfile, shapeType = 5)
    shape = s.shape(0)
    subbasinpoints = np.array(shape.points)
    subplot.add_patch(make_patch(subbasinpoints, facecolor, width = 0.8))
    extent = get_boundaries(s.shapes())

    # figure out how far one foot is on the map

    points_per_width = 72 * 8
    ft_per_km = 3280.84
    xmin, ymin, xmax, ymax = extent
    scale_factor = (points_per_width / 
                    get_distance([xmin, ymin], [xmax, ymin]) / ft_per_km)

    if catchments:
        c = Reader(cfile, shapeType = 5)

        # make patches of the catchment area

        for i in range(len(c.records())):
            catchment = c.shape(i)
            points = np.array(catchment.points)
            subplot.add_patch(make_patch(points, facecolor, width = 0.1))

    # get the flowline attributes, make an "updown" dictionary to follow flow,
    # and change the keys to comids

    f = open(flowVAAs, 'rb')
    flowlineVAAs = pickle.load(f)
    f.close()

    updown = {}
    for f in flowlineVAAs:
        if flowlineVAAs[f].down in flowlineVAAs:
            updown[flowlineVAAs[f].comid] = \
                flowlineVAAs[flowlineVAAs[f].down].comid

    flowlineVAAs = {flowlineVAAs[f].comid:flowlineVAAs[f] for f in flowlineVAAs}

    # open up and show the flowfiles

    f = Reader(flowfile, shapeType = 3)
    comid_index = f.fields.index(['COMID', 'N',  9, 0]) - 1

    all_comids = [r[comid_index] for r in f.records()]

    if flowlines:

        # get the flows and velocities from the dictionary
            
        flows = [flowlineVAAs[r[comid_index]].flow     for r in f.records()]
        vels  = [flowlineVAAs[r[comid_index]].velocity for r in f.records()]
 
        # set up an appropriate scaling (assume triangular cross-section)

        widths = [math.sqrt(4 * f / v) for f, v in zip(flows, vels)]
        
        # convert widths in feet to points on the figure; exaggerated by 3

        widths = [w * scale_factor * 3 for w in widths]

        # get the flowline and the corresponding catchment

        for i, w in zip(range(len(f.records())), widths):

            flowline = np.array(f.shape(i).points)

            # plot it

            subplot.plot(flowline[:, 0], flowline[:, 1], 'b', lw = w)

    else:
        c = Reader(combined, shapeType = 3)
        r = c.record(0)

        inlet_comid  = r[c.fields.index(['InletComID', 'N', 9, 0])  - 1]
        outlet_comid = r[c.fields.index(['OutComID', 'N', 9, 0]) - 1]

        # get the primary flowline from the hydroseqs

        comids = [inlet_comid]

        if inlet_comid in updown:
            while updown[comids[-1]] != outlet_comid:
                comids.append(updown[comids[-1]])
                if updown[comids[-1]] not in updown: break
            if outlet_comid not in comids: comids.append(outlet_comid)

        # get the flows and velocities from the dictionary

        widths = []
        for comid in comids:
            flow     = flowlineVAAs[r[comid_index]].flow
            velocity = flowlineVAAs[r[comid_index]].velocity

            # set up an appropriate scaling (assume triangular cross-section)

            widths.append(math.sqrt(4 * flow / velocity))

        # convert widths in feet to points on the figure; exaggerated by 3

        widths = [w * scale_factor * 3 for w in widths]
        
        # get the flowline and the corresponding catchment
        
        for comid, w in zip(comids, widths):

            i = all_comids.index(comid)
            flowline = np.array(f.shape(i).points)

            # plot it

            subplot.plot(flowline[:, 0], flowline[:, 1], 'b', lw = w)

    if outlets:
        f = Reader(outletfile, shapeType = 1)

        outlet_shapes = f.shapes()
        outlet_records = f.records()
        outlet_points = [o.points[0] for o in outlet_shapes]
        flow_index = f.fields.index(['AVG_FLOW', 'N', 15, 3]) - 1
        flows = [r[flow_index] for r in outlet_records]
        x1, y1 = zip(*outlet_points)
        subplot.scatter(x1, y1, marker = 'o', c = 'r', s = 40)

        if os.path.isfile(inletfile + '.shp'): 
            f = Reader(inletfile, shapeType = 1)
            inlet_shapes = f.shapes()
            inlet_points = [s.points[0] for s in inlet_shapes]
            x2, y2 = zip(*inlet_points)
            subplot.scatter(x2, y2, marker = 'o', c = 'b', s = 40)

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees', size = 13)

    # add the raster

    if raster == 'elevation':

        im = add_raster(subplot, raster_file, resolution, extent, colormap, 
                        intensity, scale = 100) 

        divider = make_axes_locatable(subplot)
        cax = divider.append_axes('right', size = 0.16, pad = 0.16)
        colorbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
        colorbar.set_label('Elevation, m', size = 12)
        cbax = pyplot.axes(colorbar.ax)
        yaxis = cbax.get_yaxis()
        ticks = yaxis.get_majorticklabels()
        for t in ticks: t.set_fontsize(10)

    else:

        pyplot.xlim([xmin, xmax])
        pyplot.ylim([ymin, ymax])

    # show it

    pyplot.tight_layout()
    pyplot.savefig(output)
    #pyplot.show()

    pyplot.close()

def plot_gage_subbasin(directory, HUC8, gage, hspfmodel, title = None, 
                       output = None, verbose = True):
    """Makes a plot of all the flowlines and catchments of a basin on top of a
    raster image file."""

    # paths to the files used herein

    sfile      = directory + '/subbasins'
    bfile      = directory + '/boundary'
    cfile      = directory + '/combined'
    flowfile   = directory + '/flowlines'
    flowVAAs   = directory + '/../../flowlineVAAs'
    dem        = directory + '/../../{0}elevations.tif'.format(HUC8)
    gagefile   = directory + '/../../{0}gagestations'.format(HUC8)

    if output is None: output = directory + '/{0}watershed'.format(gage)

    if verbose: 
        print('generating plot of watershed for gage {0}\n'.format(gage))

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # open up and show the catchments

    facecolor = (1,0,0,0.)

    b = Reader(bfile, shapeType = 5)

    points = np.array(b.shape(0).points)
    subplot.add_patch(make_patch(points, facecolor = facecolor, width = 1.))

    extent = get_boundaries(b.shapes(), space = 0.02)

    xmin, ymin, xmax, ymax = extent

    # figure out how far one foot is on the map

    points_per_width = 72 * 8
    ft_per_km = 3280.84
    scale_factor = (points_per_width / 
                    get_distance([xmin, ymin], [xmax, ymin]) / ft_per_km)

    s = Reader(sfile, shapeType = 5)

    # make patches of the subbasins

    for i in range(len(s.records())):
        shape = s.shape(i)
        points = np.array(shape.points)
        subplot.add_patch(make_patch(points, facecolor, width = 0.2))

    # get all the comids in the watershed

    f = Reader(flowfile, shapeType = 3)
    comid_index = f.fields.index(['COMID', 'N',  9, 0]) - 1

    all_comids = [r[comid_index] for r in f.records()]

    # get the flowline attributes, make an "updown" dictionary to follow flow,
    # and change the keys to comids

    with open(flowVAAs, 'rb') as f: flowlineVAAs = pickle.load(f)

    updown = {item.comid: flowlineVAAs[flowlineVAAs[key].down].comid
              for key, item in flowlineVAAs.items()
              if item.comid in all_comids}

    flowlineVAAs = {flowlineVAAs[f].comid:flowlineVAAs[f] for f in flowlineVAAs
                    if flowlineVAAs[f].comid in all_comids}

    # find the flowlines in the main channel

    c = Reader(cfile, shapeType = 3)

    comids = []
    for r in c.records():

        inlet_comid  = r[c.fields.index(['InletComID', 'N', 9, 0])  - 1]
        outlet_comid = r[c.fields.index(['OutComID', 'N', 9, 0]) - 1]

        # get the primary flowline from the hydroseqs

        comids.append(inlet_comid)

        if inlet_comid in updown:
            while updown[comids[-1]] != outlet_comid:
                comids.append(updown[comids[-1]])
                if updown[comids[-1]] not in updown: break
            if outlet_comid not in comids: comids.append(outlet_comid)

    # get the flows and velocities from the dictionary

    widths = []
    for comid in comids:
        flow     = flowlineVAAs[comid].flow
        velocity = flowlineVAAs[comid].velocity

        # estimate the flow width in feet assuming triangular 90 deg channel

        widths.append(math.sqrt(4 * flow / velocity))

    # convert widths in feet to points on the figure; exaggerated by 10

    widths = [w * scale_factor * 10 for w in widths]

    # get the flowline and the corresponding catchment

    f = Reader(flowfile, shapeType = 3)

    # show the flowlines

    for comid, w in zip(comids, widths):

        i = all_comids.index(comid)
        flowline = np.array(f.shape(i).points)

        # plot it

        subplot.plot(flowline[:, 0], flowline[:, 1], 'b', lw = w)

    # find the outlet and get the GNIS name and elevations

    i = 0
    while updown[comids[i]] in updown: i+=1
    gnis_name = f.record(all_comids.index(comids[i]))[4]

    # get the gage info

    f = Reader(gagefile, shapeType = 1)

    site_index = f.fields.index(['SITE_NO', 'C', 15, 0]) - 1
    gage_ids    = [r[site_index] for r in f.records()]
    gage_points = [g.points[0] for g in f.shapes()]

    x1, y1 = gage_points[gage_ids.index(gage)]

    subplot.scatter(x1, y1, marker = 'o', c = 'r', s = 60)

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    # add the raster

    resolution = 200
    colormap = 'gist_earth'
    intensity = None

    # use the min and max elev to set the countours

    im = add_raster(subplot, dem, resolution, extent, colormap, 
                    intensity, scale = 100) 

    divider = make_axes_locatable(subplot)
    cax = divider.append_axes('right', size = 0.16, pad = 0.16)
    colorbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
    colorbar.set_label('Elevation, m', size = 12)
    cbax = pyplot.axes(colorbar.ax)
    yaxis = cbax.get_yaxis()
    ticks = yaxis.get_majorticklabels()
    for t in ticks: t.set_fontsize(10)

    # add the title

    if not isinstance(gnis_name, bytes) > 0: descrip = ', ' + gnis_name
    else:                                    descrip = ''

    if title is None: title = 'Watershed for Gage {0}{1}'.format(gage, descrip)

    subplot.set_title(title, fontsize = 14)

    # show it

    pyplot.tight_layout()

    pyplot.savefig(output)
    #pyplot.show()

    pyplot.close()

def plot_watershed(directory, HUC8, raster = None, catchments = True, 
                   flowlines = True, outlets = False, gages = 'all',
                   dams = False, title = None, grid = False, patchcolor = None, 
                   output = None, show = False, verbose = True):
    """Makes a plot of all the flowlines and catchments of a basin on top of a
    raster image file."""

    # paths to the files used herein

    cfile      = directory + '/%s/%scatchments'         % (HUC8, HUC8)
    sfile      = directory + '/%s/%ssubbasins'          % (HUC8, HUC8)
    bfile      = directory + '/%s/%sboundaries'         % (HUC8, HUC8)
    flowfile   = directory + '/%s/%sflowlines'          % (HUC8, HUC8)
    flowVAAs   = directory + '/%s/flowlineVAAs'         %  HUC8
    dem        = directory + '/%s/%selevations.tif'     % (HUC8, HUC8)
    crop       = directory + '/%s/%scropland.tif'       % (HUC8, HUC8)
    combined   = directory + '/%s/%ssubbasin_flowlines' % (HUC8, HUC8)
    outletfile = directory + '/%s/%ssubbasin_outlets'   % (HUC8, HUC8)
    inletfile  = directory + '/%s/%ssubbasin_inlets'    % (HUC8, HUC8)
    gagefile   = directory + '/%s/%sgagestations'       % (HUC8, HUC8)

    #if output is None: 
    #    output = directory + '/%s/%sdelineated' % (HUC8, HUC8)

    if raster == 'elevation':
        raster_file = dem
        resolution = 400
        colormap = 'gist_earth'
        intensity = [100, 600]

    if raster == 'cropland':
        raster_file = crop
        resolution = 400
        colormap = 'summer_r'
        intensity = None

    if verbose: print('generating plot of watershed %s\n' % HUC8)

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # add the title

    if catchments: 
        description = 'Catchments'
        if flowlines:  description += ', Flowlines, and Gage Stations'
        else:          description += ', Major Flowlines, and Gage Stations'
    else:
        description = 'Subbasins'
        if flowlines:  description += ' and Flowlines'
        else:          description += ', Major Flowlines, and Calibration Gages'

    if raster == 'elevation': description = description + ' on 30 meter DEM'
    if raster == 'cropland':  description = description + ', Crop Land Use'

    if title is None: title = 'Cataloging Unit %s\n%s' % (HUC8, description)
    subplot.set_title(title, fontsize = 14)

    # open up and show the catchments

    if patchcolor is None: facecolor = (1,0,0,0.)
    else:                  facecolor = patchcolor

    b = Reader(bfile, shapeType = 5)

    points = np.array(b.shape(0).points)
    subplot.add_patch(make_patch(points, facecolor = facecolor, width = 1.))

    extent = get_boundaries(b.shapes(), space = 0.02)

    xmin, ymin, xmax, ymax = extent

    # figure out how far one foot is on the map

    points_per_width = 72 * 8
    ft_per_km = 3280.84
    scale_factor = (points_per_width / 
                    get_distance([xmin, ymin], [xmax, ymin]) / ft_per_km)

    if catchments:
        c = Reader(cfile, shapeType = 5)

        # make patches of the catchment area

        for i in range(len(c.records())):
            catchment = c.shape(i)
            points = np.array(catchment.points)
            subplot.add_patch(make_patch(points, facecolor, width = 0.06))

    else:
        s = Reader(sfile, shapeType = 5)

        # make patches of the subbasins

        for i in range(len(s.records())):
            shape = s.shape(i)
            points = np.array(shape.points)
            subplot.add_patch(make_patch(points, facecolor, width = 0.15))

    # get the flowline attributes, make an "updown" dictionary to follow flow,
    # and change the keys to comids

    f = open(flowVAAs, 'rb')
    flowlineVAAs = pickle.load(f)
    f.close()

    updown = {}
    for f in flowlineVAAs:
        if flowlineVAAs[f].down in flowlineVAAs:
            updown[flowlineVAAs[f].comid] = \
                flowlineVAAs[flowlineVAAs[f].down].comid

    flowlineVAAs = {flowlineVAAs[f].comid:flowlineVAAs[f] for f in flowlineVAAs}

    # open up and show the flowfiles

    f = Reader(flowfile, shapeType = 3)
    comid_index = f.fields.index(['COMID', 'N',  9, 0]) - 1

    all_comids = [r[comid_index] for r in f.records()]

    if flowlines:     # show all the flowfiles 
        
        # get the flows and velocities from the dictionary
        
        widths = []
        comids = []
        for comid in all_comids:
            if comid in flowlineVAAs:
                flow = flowlineVAAs[comid].flow
                velocity = flowlineVAAs[comid].velocity

                # estimate flow width (ft) assuming triangular 90 deg channel 

                comids.append(comid)
                widths.append(math.sqrt(4 * flow / velocity))
        
        # convert widths in feet to points on the figure; exaggerated by 10

        widths = [w * scale_factor * 10 for w in widths]

        # get the flowline and the corresponding catchment

        for comid, w in zip(comids, widths):

            i = all_comids.index(comid)
            flowline = np.array(f.shape(i).points)

            # plot it

            subplot.plot(flowline[:, 0], flowline[:, 1], 'b', lw = w)

    else: # use the combined flowlines

        c = Reader(combined, shapeType = 3)

        comids = []
        for r in c.records():

            inlet_comid  = r[c.fields.index(['InletComID', 'N', 9, 0])  - 1]
            outlet_comid = r[c.fields.index(['OutComID', 'N', 9, 0]) - 1]

            # get the primary flowline from the hydroseqs

            comids.append(inlet_comid)

            if inlet_comid in updown:
                while updown[comids[-1]] in updown:
                    comids.append(updown[comids[-1]])
                if outlet_comid not in comids: comids.append(outlet_comid)

        # get the flows and velocities from the dictionary

        widths = []
        for comid in comids:
            flow     = flowlineVAAs[comid].flow
            velocity = flowlineVAAs[comid].velocity

            # estimate the flow width in feet assuming triangular 90 deg channel

            widths.append(math.sqrt(4 * flow / velocity))
                
        # convert widths in feet to points on the figure; exaggerated by 10

        widths = [w * scale_factor * 10 for w in widths]

        # get the flowline and the corresponding catchment

        for comid, w in zip(comids, widths):

            i = all_comids.index(comid)
            flowline = np.array(f.shape(i).points)

            # plot it

            subplot.plot(flowline[:, 0], flowline[:, 1], 'b', lw = w)

    if outlets:

        f = Reader(outletfile, shapeType = 1)

        outlet_shapes = f.shapes()
        outlet_records = f.records()
        flow_index = f.fields.index(['AVG_FLOW', 'N', 15, 3]) - 1
        flows = [r[flow_index] for r in outlet_records]
        outlet_points = [o.points[0] for o in outlet_shapes]
        x1, y1 = zip(*outlet_points)
        subplot.scatter(x1, y1, marker = 'o', c = 'r', s = 30)

        if os.path.isfile(inletfile + '.shp'): 
            f = Reader(inletfile, shapeType = 1)
            inlet_shapes = f.shapes()
            inlet_points = [s.points[0] for s in inlet_shapes]
            inlet_flows = [r[flow_index] for r in f.records()]
            x2, y2 = zip(*inlet_points)
            subplot.scatter(x2, y2, marker = 'o', c = 'b', s = 30)

    if gages == 'all':
        
        f = Reader(gagefile, shapeType = 1)

        gage_shapes = f.shapes()
        gage_points = [g.points[0] for g in gage_shapes]

        x1, y1 = zip(*gage_points)
        subplot.scatter(x1, y1, marker = 'o', c = 'r', s = 30)

    elif gages == 'calibration': # show gages used for calibration

        f1 = Reader(outletfile, shapeType = 1)

        outlet_records = f1.records()

        site_index = f1.fields.index(['SITE_NO', 'C', 15, 0]) - 1
        flow_index = f1.fields.index(['AVG_FLOW', 'N', 15, 3]) - 1

        flows = [r[flow_index] for r in outlet_records]
        sites = [r[site_index] for r in outlet_records]

        f2 = Reader(gagefile, shapeType = 1)

        gage_shapes  = f2.shapes()
        gage_records = f2.records()

        site_index = f2.fields.index(['SITE_NO', 'C', 15, 0]) - 1

        gage_points = []
        for shape, record in zip(gage_shapes, gage_records):

            if record[site_index] in sites:

                gage_points.append(shape.points[0])

        x1, y1 = zip(*gage_points)
        subplot.scatter(x1, y1, marker = 'o', c = 'r', s = 30)

    if dams:  # show dams

        f = Reader(outletfile, shapeType = 1)

        outlet_records = f.records()

        dam_index = f.fields.index(['NIDID', 'C', 7, 0]) - 1

        nidids = [r[dam_index] for r in outlet_records]

        dam_points = []
        for nid, s in zip(nidids, f.shapes()):
            if isinstance(nid, bytes): nid = nid.decode('utf-8')
            nid = nid.strip()

            if len(nid) > 0:
                
                dam_points.append(s.points[0])

        x1, y1 = zip(*dam_points)
        subplot.scatter(x1, y1, marker = 's', c = 'y', s = 30)

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    # add the raster

    if raster is 'elevation':

        im = add_raster(subplot, raster_file, resolution, extent, colormap, 
                        intensity, scale = 100) 

        divider = make_axes_locatable(subplot)
        cax = divider.append_axes('right', size = 0.16, pad = 0.16)
        colorbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
        colorbar.set_label('Elevation, m', size = 12)
        cbax = pyplot.axes(colorbar.ax)
        yaxis = cbax.get_yaxis()
        ticks = yaxis.get_majorticklabels()
        for t in ticks: t.set_fontsize(10)

    elif raster is 'cropland':

        im = add_scatter(subplot, raster_file, subbasinpoints)

    else:

        pyplot.xlim([xmin, xmax])
        pyplot.ylim([ymin, ymax])

    if grid:

        subplot.xaxis.set_minor_locator(MultipleLocator(0.1))
        subplot.yaxis.set_minor_locator(MultipleLocator(0.1))

        subplot.xaxis.grid(True, 'minor', linestyle = '-', linewidth = 0.5)
        subplot.yaxis.grid(True, 'minor', linestyle = '-', linewidth = 0.5)

    # show it

    pyplot.tight_layout()

    if output is not None:  pyplot.savefig(output)

    if show: pyplot.show()

    pyplot.close()

def plot_catchment(catchment, flowline, elevation, cropland):
    """Plots a catchment and its flowline on a raster image."""

    elev_raster = (' on 30 meter DEM', elevation, 200, 'gist_earth', None) 
    crop_raster = (', Crop Land Use',  cropland,  200, 'summer_r',   None) 

    shape_list = format_shape(catchment.points)

    extent = get_boundaries([catchment])
    xmin, ymin, xmax, ymax = extent

    gs = gridspec.GridSpec(2,2)

    # add the catchment and flowline geometry

    subplot1 = pyplot.subplot(gs[0, 0], aspect = 'equal')
    subplot1.set_title('Catchment and Flowline on DEM', size = 12)

    # make a patch of the catchment area

    for shape in shape_list:
        subplot1.add_patch(make_patch(shape, facecolor = 'None'))

    # add the flowline

    xs = [p[0] for p in flowline.points]
    ys = [p[1] for p in flowline.points]

    subplot1.plot(xs, ys, 'b')

    # set up the axes

    subplot1.set_xlim(xmin, xmax)
    subplot1.set_ylim(ymin, ymax)
    subplot1.set_xlabel('Longitude', fontsize = 12)
    subplot1.set_ylabel('Latitude',  fontsize = 12)

    # add the raster

    d, f, resolution, colormap, intensity = elev_raster
    im1 = add_raster(subplot1, f, resolution, extent, colormap, intensity) 
    divider1  = make_axes_locatable(subplot1)
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    colorbar1 = pyplot.colorbar(im1, cax = cax1,
                                orientation = 'vertical')
    colorbar1.set_label('Elevation, cm', fontsize = 11)
    pyplot.axes(colorbar1.ax)
    pyplot.yticks(fontsize = 10)

    # look at the DEM raster

    xs, ys, zs = get_raster_in_poly(f, catchment.points, verbose = verbose)

    intensity = [min(zs), max(zs)]

    subplot2 = pyplot.subplot(gs[1, 0], aspect = 'equal')
    subplot2.set_title('DEM Raster', size = 12)

    norm = colors.Normalize(vmin = intensity[0], vmax = intensity[1])
    raster_plot2 = subplot2.scatter(xs, ys, c = zs, cmap = colormap, 
                                    norm = norm)

    divider2  = make_axes_locatable(subplot2)
    cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
    colorbar2 = pyplot.colorbar(raster_plot2, cax = cax2,
                                orientation = 'vertical')
    colorbar2.set_label('Elevation, cm', fontsize = 11)
    pyplot.axes(colorbar2.ax)
    pyplot.yticks(fontsize = 10)

    subplot2.set_xlim(xmin, xmax)
    subplot2.set_ylim(ymin, ymax)
    subplot2.set_xlabel('Longitude', fontsize = 12)
    subplot2.set_ylabel('Latitude',  fontsize = 12)

    # look at the bounding raster

    flow_zs = get_raster(f, np.array(flowline.points))
        
    flowpoints = (np.array([p[0]  for p in flowline.points]),
                  np.array([p[1]  for p in flowline.points]),
                  np.array([flow_zs[j] for j in range(len(flowline.points))]))

    subplot3 = pyplot.subplot(gs[0, 1], aspect = 'equal')
    subplot3.set_title('Catchment Boundary and Flowline', size = 12)

    catchpoints = get_raster_on_poly(f, catchment.points, verbose = verbose)

    xs, ys, zs = zip(*catchpoints)

    flowplot = subplot3.scatter(flowpoints[0], flowpoints[1], 
                                c = flowpoints[2], cmap = colormap, norm = norm)
    raster_plot3 = subplot3.scatter(xs, ys, c = zs, cmap = colormap, 
                                    norm = norm)

    divider3  = make_axes_locatable(subplot2)
    cax3 = divider3.append_axes('right', size = '5%', pad = 0.05)
    colorbar3 = pyplot.colorbar(raster_plot3, cax = cax3,
                                orientation = 'vertical')
    colorbar3.set_label('Elevation, cm', fontsize = 11)
    pyplot.axes(colorbar3.ax)
    pyplot.yticks(fontsize = 10)

    subplot3.set_xlim(xmin, xmax)
    subplot3.set_ylim(ymin, ymax)
    subplot3.set_xlabel('Longitude', fontsize = 12)
    subplot3.set_ylabel('Latitude',  fontsize = 12)

    # look at the crop raster

    d, f, resolution, colormap, intensity = crop_raster
    xs, ys, zs = get_raster_in_poly(f, catchment.points, verbose = verbose)

    intensity = [min(zs), max(zs)]

    subplot4 = pyplot.subplot(gs[1, 1], aspect = 'equal')
    subplot4.set_title('Cropland Raster', size = 12)

    norm = colors.Normalize(vmin = intensity[0], vmax = intensity[1])
    raster_plot4 = subplot4.scatter(xs, ys, c = zs, cmap = colormap, 
                                    norm = norm)

    divider4  = make_axes_locatable(subplot4)
    cax4 = divider4.append_axes('right', size = '5%', pad = 0.05)
    colorbar4 = pyplot.colorbar(raster_plot4, cax = cax4, 
                                orientation = 'vertical')
    colorbar4.set_label('Elevation, cm', fontsize = 11)
    pyplot.axes(colorbar4.ax)
    pyplot.yticks(fontsize = 10)

    subplot4.set_xlim(xmin, xmax)
    subplot4.set_ylim(ymin, ymax)
    subplot4.set_xlabel('Longitude', fontsize = 12)
    subplot4.set_ylabel('Latitude',  fontsize = 12)

    # show it

    pyplot.tight_layout()

    pyplot.show()

    pyplot.close()

def plot_gage_segments(directory, HUC8, gage, year, aggregatefile, 
                       output, datatype = 'raw', overwrite = False,
                       pixels = 1000, verbose = True):
    """Makes a plot of all the flowlines and catchments of a basin on top of a
    raster image file."""

    color_codes = [(255, 255, 255),  # empty
                   (255, 211,   0),  # corn 
                   ( 38, 112,   0),  # soybeans
                   (255, 221, 165),  # other grain
                   (155, 155, 155),  # developed
                   ( 76, 112, 163),  # water/wetland
                   (147, 204, 147),  # forest
                   (255, 165, 226),  # hay/alfalfa
                   (191, 191, 119),  # fallow
                   (232, 255, 191),  # pasture/grass
                   (  0,   0,   0)]  # other

    folder    = '{0}/{1}/calibrations/{2}'.format(directory, HUC8, gage)
    boundary  = '{0}/boundary'.format(folder)
    shapefile = '{0}/subbasins'.format(folder)
    landfile  = directory + '/%s/landuse/%dlanduse' % (HUC8, year) 
    summary   = directory + '/%s/%slanduse.csv'     % (HUC8, HUC8)
    geotiff   = directory + '/%s/%scropland%d.tif'  % (HUC8, HUC8, year)

    if os.path.isfile(output):
        if not overwrite:
            if verbose: print('file %s exists\n' % output)
            return
        elif verbose: print('warning: overwriting %s' % output)

    if verbose: print('generating land use plot for {}'.format(year))

    # make the figure

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 11)

    # add the title

    if   datatype == 'subbasins': 
        title = 'HSPF Land Segments for Gage {0}'.format(gage)
    else:                       
        title = 'Land Use Data from {0} for Gage {1}'.format(year, gage)

    subplot.set_title(title, size = 14)

    # open the boundary shapefile and draw the boundary

    s      = Reader(boundary, shapeType = 5)
    b      = np.array(s.shape(0).points)
    bbox   = s.shape(0).bbox
    bbox   = [bbox[0], bbox[2], bbox[1], bbox[3]]
    extent = get_boundaries(s.shapes(), space = 0.02)

    # open up the aggregate dictionary to map values

    m, landtypes, groups = get_aggregate_map(aggregatefile)

    # get the land use fraction for each category for the year

    if datatype == 'subbasins': 

        # open up the shapefile

        s = Reader(shapefile, shapeType = 5)

        comid_index = s.fields.index(['ComID', 'N',  9, 0]) - 1

        n = len(s.records())

        # set up the image grid using the extents for the entire watershed

        xs, ys = zip(*b)

        # get the pixel width and origin

        w = (max(xs) - min(xs)) / pixels
        x0, y0 = min(xs), min(ys)

        # calculate the image array height and the height of a pixel

        height = math.ceil((max(ys) - min(ys)) / (max(xs) - min(xs)) * pixels)
        h = (max(ys) - min(ys)) / height

        # set up the image array

        image_array = np.zeros((height, pixels), dtype = 'uint8')

        for i in range(n):
            shape = s.shape(i)
            comid = s.record(i)[comid_index]
            points = np.array(shape.points)

            # convert the shape to pixel coordinates

            pixel_polygon = [(get_pixel(x, x0, w), get_pixel(y, y0, h))
                             for x, y in points]
           
            # make a PIL image with the appropriate dimensions to use as a mask 

            rasterpoly = Image.new('L', (pixels, height), 1)
            rasterize  = ImageDraw.Draw(rasterpoly)

            # rasterize the polygon

            rasterize.polygon(pixel_polygon, 0)

            # convert the PIL array to numpy boolean to use as a mask

            mask = 1 - np.array(rasterpoly)

            # get the total number of pixels in the shape

            tot = mask.sum()

            # iterate from left to right and get the fraction
            # inside as a function of x

            fractions = [column.sum() / tot for column in mask.transpose()]

            area_cdf = [0]
            for f in fractions[1:]: area_cdf.append(area_cdf[-1] + f)

            # open up the landuse data file

            with open(landfile, 'rb') as f: subbasin_landuse = pickle.load(f)

            codes, areas = subbasin_landuse['{}'.format(comid)]

            # aggregate the areas

            group_areas = {g:0. for g in groups}
            for code, area in zip(codes, areas):
                group = m[code]
                group_areas[group] +=  area

            # convert the areas to fractions

            fractions = [group_areas[group] / sum(list(group_areas.values())) 
                         for group in group_areas]

            # convert the land use fractions into a land use cdf

            land_cdf = [fractions[0]]
            for f in fractions[1:]: land_cdf.append(land_cdf[-1] + f)

            # use the area cdf to determine the breaks in land use patches

            color_array = np.zeros(len(mask[0]), dtype = 'uint8')
            i = 0
            for p, group in zip(land_cdf[:-1], groups):
                while area_cdf[i] <= p: 
                    color_array[i] = group
                    if i < len(area_cdf) - 1: i += 1
                    else: break
            color_array[i:] = groups[-1]

            # multiply the color band array by the mask to get the subbasin img

            sub_img = mask * color_array

            # add the new mask to the watershed image

            image_array = image_array + sub_img

            # add a patch for the shape boundary

            subplot.add_patch(make_patch(points, (1,0,0,0), width = 0.5))

    elif datatype == 'all':

        # open up the summary csv and read the cumulative data

        with open(summary, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader][3:]

        # re-organize into columns (ignore the rest)

        years   = [int(y) for y in rows[0][2:]]
        columns = [column for column in zip(*rows[1:1 + len(groups)])]
        data    = columns[2:]

        # pick out the areas for the year of interest and convert fraction

        areas = [float(v) for v in data[years.index(year)]]
        fractions = [a / sum(areas) for a in areas]

        xmin, ymin, xmax, ymax = s.shape(0).bbox

        # get the cdf for the area of the shape

        xs, area_cdf = poly_to_cdf(b, dim = 'x', n = pixels)

        # convert the land use fractions into a land use cdf

        land_cdf = [fractions[0]]
        for f in fractions[1:]: land_cdf.append(land_cdf[-1] + f)

        # use the area cdf to determine the breaks between the land use patches

        i = 0
        indices = [0]
        for p in land_cdf[:-1]:
            while area_cdf[i] < p: i+=1
            indices.append(i)
            i+=1
        if indices[-1] != len(xs) - 1: indices.append(len(xs) - 1)

        # set up a list of the patches and get another mask for the polygon

        mask = shape_to_mask(b, pixels)

        # set up an array for the image colors

        color_array = np.zeros(len(mask[0]), dtype = 'uint8')
        for i, j, code in zip(indices[:-1], indices[1:], groups): 
            color_array[i:j] = code
            color_array[i] = code

        # multiply the mask by the landcode and link the landcodes to colors

        image_array = (mask * color_array)

    else:

        # open up the cropland raster file

        raw, origin = get_raster_in_poly(geotiff, b, dtype = 'uint8', verbose =
                                         False)

        values = np.unique(raw)

        # map the raw data to the HSPF landuse

        image_array = np.zeros((len(raw), len(raw[0])), dtype = 'uint8')

        m[0] = 0
        for v in values:
            image_array[np.where(raw == v)] = m[v]
       
        image_array = image_array[::2,::2]

    # set up a custom color map

    color_table = [(r / 255, g / 255, b / 255) for r, g, b in color_codes]
    cmap = colors.ListedColormap(color_table)
    bounds = [0] + list(landtypes.keys()) + [len(landtypes) + 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # adjust the plot appropriately

    xmin, ymin, xmax, ymax = extent

    im = subplot.imshow(image_array, extent = bbox, origin = 'upper left', 
                        interpolation = 'nearest', cmap = cmap, norm = norm);

    # add a patch for the watershed boundary 

    subplot.add_patch(make_patch(b, (1,0,0,0), width = 1))

    # add the legend using a dummy box to make patches for the legend

    dummybox = [[0,0], [0,1], [1,1], [1,0], [0,0]]
    handles, labels = [], []
    for code, color in zip(landtypes, color_table[1:]):
        handles.append(subplot.add_patch(make_patch(dummybox, facecolor = color,
                                                    width = 0)))
        labels.append(landtypes[code])

    leg = subplot.legend(handles, labels, bbox_to_anchor = (1.0, 0.5), 
                         loc = 'center left', title = 'Land Use Categories')
    legtext = leg.get_texts()
    pyplot.setp(legtext, fontsize = 10)
    subplot.set_position([0.125, 0.1, 0.6, 0.8])

    # add the labels and set the limits

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    pyplot.xlim([xmin, xmax])
    pyplot.ylim([ymin, ymax])

    # show it

    pyplot.savefig(output)

    pyplot.clf()
    #pyplot.show()
    pyplot.close()

def plot_segments(directory, HUC8, year, aggregatefile, color_codes,
                  output, datatype = 'raw', overwrite = False,
                  pixels = 1000, verbose = True, vverbose = False):
    """Makes a plot of all the flowlines and catchments of a basin on top of a
    raster image file."""

    boundary  = directory + '/%s/%sboundaries'      % (HUC8, HUC8)
    shapefile = directory + '/%s/%ssubbasins'       % (HUC8, HUC8)
    landfile  = directory + '/%s/landuse/%dlanduse' % (HUC8, year) 
    summary   = directory + '/%s/%slanduse.csv'     % (HUC8, HUC8)
    geotiff   = directory + '/%s/%scropland%d.tif'  % (HUC8, HUC8, year)

    if os.path.isfile(output):
        if not overwrite:
            if verbose: print('file %s exists' % output)
            return
        elif vverbose: print('warning: overwriting %s' % output)

    if verbose: print('generating land use plot for %d\n' % year)

    # make the figure

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 11)

    # add the title

    if   datatype == 'subbasins': plottype = ['HSPF', 'Segments']
    elif datatype == 'all':       plottype = ['Watershed Averaged', 'Segments']
    else:                         plottype = ['Filtered Raw', 'Data']

    description = '%d %s Land Use %s' % (year, plottype[0], plottype[1])
    title = ('Cataloging Unit %s\n%s' % (HUC8, description))
    subplot.set_title(title, size = 14)

    # open the boundary shapefile and draw the boundary

    s      = Reader(boundary, shapeType = 5)
    b      = np.array(s.shape(0).points)
    bbox   = s.shape(0).bbox
    bbox   = [bbox[0], bbox[2], bbox[1], bbox[3]]
    extent = get_boundaries(s.shapes(), space = 0.02)

    # open up the aggregate dictionary to map values

    m, landtypes, groups = get_aggregate_map(aggregatefile)

    # get the land use fraction for each category for the year

    if datatype == 'subbasins': 

        # open up the shapefile

        s = Reader(shapefile, shapeType = 5)

        comid_index = s.fields.index(['ComID', 'N',  9, 0]) - 1

        n = len(s.records())

        # set up the image grid using the extents for the entire watershed

        xs, ys = zip(*b)

        # get the pixel width and origin

        w = (max(xs) - min(xs)) / pixels
        x0, y0 = min(xs), min(ys)

        # calculate the image array height and the height of a pixel

        height = math.ceil((max(ys) - min(ys)) / (max(xs) - min(xs)) * pixels)
        h = (max(ys) - min(ys)) / height

        # set up the image array

        image_array = np.zeros((height, pixels), dtype = 'uint8')

        for i in range(n):
            shape = s.shape(i)
            comid = '{}'.format(s.record(i)[comid_index])
            points = np.array(shape.points)

            # convert the shape to pixel coordinates

            pixel_polygon = [(get_pixel(x, x0, w), get_pixel(y, y0, h))
                             for x, y in points]
           
            # make a PIL image with the appropriate dimensions to use as a mask 

            rasterpoly = Image.new('L', (pixels, height), 1)
            rasterize  = ImageDraw.Draw(rasterpoly)

            # rasterize the polygon

            rasterize.polygon(pixel_polygon, 0)

            # convert the PIL array to numpy boolean to use as a mask

            mask = 1 - np.array(rasterpoly)

            # get the total number of pixels in the shape

            tot = mask.sum()

            # iterate from left to right and get the fraction
            # inside as a function of x

            fractions = [column.sum() / tot for column in mask.transpose()]

            area_cdf = [0]
            for f in fractions[1:]: area_cdf.append(area_cdf[-1] + f)

            # open up the landuse data file

            with open(landfile, 'rb') as f: subbasin_landuse = pickle.load(f)

            codes, areas = subbasin_landuse[comid]

            # aggregate the areas

            group_areas = {g:0. for g in groups}
            for code, area in zip(codes, areas):
                group = m[code]
                group_areas[group] +=  area

            # convert the areas to fractions

            fractions = [group_areas[group] / sum(list(group_areas.values())) 
                         for group in group_areas]

            # convert the land use fractions into a land use cdf

            land_cdf = [fractions[0]]
            for f in fractions[1:]: land_cdf.append(land_cdf[-1] + f)

            # use the area cdf to determine the breaks in land use patches

            color_array = np.zeros(len(mask[0]), dtype = 'uint8')
            i = 0
            for p, group in zip(land_cdf[:-1], groups):
                while area_cdf[i] <= p: 
                    color_array[i] = group
                    if i < len(area_cdf) - 1: i += 1
                    else: break
            color_array[i:] = groups[-1]

            # multiply the color band array by the mask to get the subbasin img

            sub_img = mask * color_array

            # add the new mask to the watershed image

            image_array = image_array + sub_img

            # add a patch for the shape boundary

            subplot.add_patch(make_patch(points, (1,0,0,0), width = 0.5))

    elif datatype == 'all':

        # open up the summary csv and read the cumulative data

        with open(summary, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader][3:]

        # re-organize into columns (ignore the rest)

        years   = [int(y) for y in rows[0][2:]]
        columns = [column for column in zip(*rows[1:1 + len(groups)])]
        data    = columns[2:]

        # pick out the areas for the year of interest and convert fraction

        areas = [float(v) for v in data[years.index(year)]]
        fractions = [a / sum(areas) for a in areas]

        xmin, ymin, xmax, ymax = s.shape(0).bbox

        # get the cdf for the area of the shape

        xs, area_cdf = poly_to_cdf(b, dim = 'x', n = pixels)

        # convert the land use fractions into a land use cdf

        land_cdf = [fractions[0]]
        for f in fractions[1:]: land_cdf.append(land_cdf[-1] + f)

        # use the area cdf to determine the breaks between the land use patches

        i = 0
        indices = [0]
        for p in land_cdf[:-1]:
            while area_cdf[i] < p: i+=1
            indices.append(i)
            i+=1
        if indices[-1] != len(xs) - 1: indices.append(len(xs) - 1)

        # set up a list of the patches and get another mask for the polygon

        mask = shape_to_mask(b, pixels)

        # set up an array for the image colors

        color_array = np.zeros(len(mask[0]), dtype = 'uint8')
        for i, j, code in zip(indices[:-1], indices[1:], groups): 
            color_array[i:j] = code
            color_array[i] = code

        # multiply the mask by the landcode and link the landcodes to colors

        image_array = (mask * color_array)

    else:

        # open up the cropland raster file

        raw, origin = get_raster_in_poly(geotiff, b, dtype = 'uint8', verbose =
                                         False)

        values = np.unique(raw)

        # map the raw data to the HSPF landuse

        image_array = np.zeros((len(raw), len(raw[0])), dtype = 'uint8')

        m[0] = 0
        for v in values:
            image_array[np.where(raw == v)] = m[v]
       
        image_array = image_array[::2,::2]

    # set up a custom color map

    color_table = [(r / 255, g / 255, b / 255) for r, g, b in color_codes]
    cmap = colors.ListedColormap(color_table)
    bounds = [0] + list(landtypes.keys()) + [len(landtypes) + 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # adjust the plot appropriately

    xmin, ymin, xmax, ymax = extent

    im = subplot.imshow(image_array, extent = bbox, origin = 'upper left', 
                        interpolation = 'nearest', cmap = cmap, norm = norm);

    # add a patch for the watershed boundary 

    subplot.add_patch(make_patch(b, (1,0,0,0), width = 1))

    # add the legend using a dummy box to make patches for the legend

    dummybox = [[0,0], [0,1], [1,1], [1,0], [0,0]]
    handles, labels = [], []
    for code, color in zip(landtypes, color_table[1:]):
        handles.append(subplot.add_patch(make_patch(dummybox, facecolor = color,
                                                    width = 0)))
        labels.append(landtypes[code])

    leg = subplot.legend(handles, labels, bbox_to_anchor = (1.0, 0.5), 
                         loc = 'center left', title = 'Land Use Categories')
    legtext = leg.get_texts()
    pyplot.setp(legtext, fontsize = 10)
    subplot.set_position([0.125, 0.1, 0.6, 0.8])

    # add the labels and set the limits

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    pyplot.xlim([xmin, xmax])
    pyplot.ylim([ymin, ymax])

    # show it

    pyplot.savefig(output)
    #pyplot.show()
    pyplot.clf()
    pyplot.close()

def landuse_plots(directory, HUC8, years, aggregate, pixels = 2000, 
                  overwrite = False, verbose = True, fmt = 'png'):
    """Makes plots of the raw landuse data, the watershed averaged data, and
    the subbasin averaged data for an 8-digit watershed."""

    # set up a color scheme for the land use categories

    color_codes = [(255, 255, 255),  # empty
                   (255, 211,   0),  # corn 
                   ( 38, 112,   0),  # soybeans
                   (255, 221, 165),  # other grain
                   (155, 155, 155),  # developed
                   ( 76, 112, 163),  # water/wetland
                   (147, 204, 147),  # forest
                   (255, 165, 226),  # hay/alfalfa
                   (191, 191, 119),  # fallow
                   (232, 255, 191),  # pasture/grass
                   (  0,   0,   0)]  # other

    for year in years:
    
        raw       = (directory + '/%s/images/%slanduse%d_raw.%s' %      
                     (HUC8, HUC8, year, fmt))
        a         = (directory + '/%s/images/%slanduse%d_all.%s' % 
                     (HUC8, HUC8, year, fmt))
        subbasins = (directory + '/%s/images/%slanduse%d_subbasins.%s' % 
                     (HUC8, HUC8, year, fmt))
        try:

            plot_segments(directory, HUC8, year, aggregate, color_codes, raw,
                          datatype = 'raw', overwrite = overwrite,
                          pixels = pixels, verbose = verbose)
        except MemoryError: 
            if verbose: print('warning: unable to generate raw file\n')

        try:
            plot_segments(directory, HUC8, year, aggregate, color_codes, a,
                          datatype = 'all', overwrite = overwrite,
                          pixels = pixels, verbose = verbose)
        except MemoryError:
            plot_segments(directory, HUC8, year, aggregate, color_codes, 
                          datatype = 'all', output = a, overwrite = overwrite,
                          pixels = pixels // 2, verbose = verbose)
        except:  
            if verbose: print('warning: unable to generate landuse plot\n')

        try:
            plot_segments(directory, HUC8, year, aggregate, color_codes, 
                          subbasins, datatype = 'subbasins', overwrite = 
                          overwrite, pixels = pixels, verbose = verbose)
        except MemoryError:
            plot_segments(directory, HUC8, year, aggregate, color_codes, 
                          datatype = 'subbasins', output = subbasins, 
                          overwrite = overwrite, pixels = pixels // 2,
                          verbose = verbose)
        except:  
            if verbose: print('warning: unable to generate landuse plot')

def plot_aquifers(directory, HUC8, subbasins = True, output = None, 
                  verbose = True):
    """Makes a plot of the aquifers in a watershed."""

    # colors for aquifers based on the AQ_CODE

    colors = {101: (0,0,0),
              102: (0,0,0),
              103: (0,0,0),
              104: (0,0,0),
              105: (0,0,0),
              106: (0,0,0),
              107: (0,0,0),
              108: (0,0,0),
              109: (0,0,0),
              110: (0,0,0),
              111: (0,0,0),
              112: (0,0,0),
              114: (0,0,0),
              115: (0,0,0),
              116: (0,0,0),
              117: (0,0,0),
              201: (0,0,0),
              202: (0,0,0),
              203: (0,0,0),
              204: (0,0,0),
              205: (0,0,0),
              301: (0,0,0),
              302: (0,0,0),
              304: (0,0,0),
              305: (0,0,0),
              306: (0,0,0),
              307: (0,0,0),
              308: (0,0,0),
              309: (0,0,0),
              310: (0,0,0),
              311: (0,0,0),
              312: (0.5,0.5,0),
              313: (0,0,0),
              314: (0,0,0),
              315: (0,0,0),
              316: (0,0,0),
              401: (0,0,0),
              402: (0,0,0),
              405: (0,0,0),
              406: (0,0,0),
              407: (0,0,0),
              410: (0,0.6,0),
              411: (0,0,0),
              412: (0,0,0),
              413: (0,0,0),
              414: (0,0,0),
              416: (0,0,0),
              417: (0,0,0),
              418: (0,0,0),
              419: (0,0,0),
              420: (0,0,0),
              501: (0,0,0),
              502: (0,0,0),
              503: (0,0.5,1),
              504: (0,0,0),
              505: (0,0,0),
              601: (0,0,0),
              606: (0,0,0),
              607: (0,0,0),
              608: (0,0,0),
              609: (0,0,0),
              610: (0,0,0),
              611: (0,0,0),
              999: (0.8,0.8,0.8)
              }

    # paths to the files used herein

    sfile = directory + '/%s/%ssubbasins'  % (HUC8, HUC8)
    bfile = directory + '/%s/%sboundaries' % (HUC8, HUC8)
    afile = directory + '/%s/%saquifers' % (HUC8, HUC8)

    if output is None: 
        output = directory + '/%s/%saquifers' % (HUC8, HUC8)

    if verbose: print('generating plot of aquifers in watershed %s\n' % HUC8)

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # add the title

    title = 'Cataloging Unit %s Aquifers' % HUC8
    subplot.set_title(title, fontsize = 14)

    # open up and show the aquifers

    a = Reader(afile, shapeType = 5)

    name_index   = a.fields.index(['ROCK_NAME', 'C', 40, 0]) - 1
    type_index   = a.fields.index(['ROCK_TYPE', 'N',  9, 0]) - 1
    aq_index     = a.fields.index(['AQ_NAME',   'C', 60, 0]) - 1
    number_index = a.fields.index(['AQ_CODE',   'N',  9, 0]) - 1

    # make a dictionary to keep track of aquifer types and colors

    aquifertypes = {}
    for r in a.records():
        if r[number_index] not in aquifertypes:
            aquifertypes[r[number_index]] = r[aq_index]

    # draw the "no aquifer patch" first

    points = a.shape(len(a.records()) - 1).points
    record = a.record(len(a.records()) - 1)
    subplot.add_patch(make_patch(points, colors[record[number_index]], 
                                 width = 0))

    for i in range(len(a.records()) - 2):
        shape  = a.shape(i)
        record = a.record(i)
        points = shape.points
        parts  = shape.parts
        color  = colors[record[number_index]]

        if len(parts) > 2:
            for i, j in zip(parts[:-1], parts[1:]):
                subplot.add_patch(make_patch(points[i:j], color, width = 0))
        else:
            subplot.add_patch(make_patch(points, color, width = 0))

    # open up and show the boundary

    b = Reader(bfile, shapeType = 5)

    points = np.array(b.shape(0).points)
    subplot.add_patch(make_patch(points, facecolor = (1,0,0,0.), width = 1.))

    xmin, ymin, xmax, ymax = get_boundaries(b.shapes(), space = 0.02)

    # make patches of the subbasins

    if subbasins:
        s = Reader(sfile, shapeType = 5)

        for i in range(len(s.records())):
            shape = s.shape(i)
            points = np.array(shape.points)
            subplot.add_patch(make_patch(points, (1,0,0,0.), width = 0.15))

    # add the legend using a dummy box to make patches for the legend

    dummybox = [[0,0], [0,1], [1,1], [1,0], [0,0]]
    handles, labels = [], []
    for code in aquifertypes.keys():
        color = colors[code]
        handles.append(subplot.add_patch(make_patch(dummybox, facecolor = color,
                                                    width = 0)))
        labels.append(aquifertypes[code])

    leg = subplot.legend(handles, labels, bbox_to_anchor = (1.0, 0.5), 
                         loc = 'center left', title = 'Aquifer Types')
    legtext = leg.get_texts()
    pyplot.setp(legtext, fontsize = 8)
    subplot.set_position([0.1, 0.1, 0.6, 0.75])

    # adjust the extent

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    pyplot.xlim([xmin, xmax])
    pyplot.ylim([ymin, ymax])

    # show it

    pyplot.savefig(output)
    #pyplot.show()

    pyplot.close()

def plot_mass_flow(watershed, output, fontsize = 6, theight = 0.2, l = 8.5, 
                   w = 11, verbose = True, overwrite = True):
    """Makes a schematic of the mass linkages between the various subbasins
    in a watershed.
    """

    if os.path.exists(output) and not overwrite:
        if verbose: print('file %s exists' % output)
        return
    elif verbose: print('generating a mass linkage plot\n')

    fontheight = fontsize / 72.
    rheight = 3 * fontheight
    rwidth  = 12 * fontheight
    xgap = fontheight
    ygap = rheight
    awidth = rheight / 4
    aheight = rheight / 3

    title = 'Subbasin Reach Mass Flow Diagram'

    # set up a sheet to write the image

    fig = pyplot.figure(figsize = (w, l))

    ax  = fig.add_subplot(111, xlim = (0, w), ylim = (l, 0), 
                          aspect = 'equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    t = ax.set_title(title)

    # divide the subbasins into rows and put them on the chart
    # start at the bottom to organize the linkages better

    rows = [watershed.outlets, ['outlet']]

    top = False
    while not top:
        row = []
        for next in rows[0]:
            for subbasin in watershed.updown:
                if watershed.updown[subbasin] == next: row.append(subbasin)
        if len(row) > 0: rows.insert(0, row)
        else: top = True

    # add an inlet box in the row above each inlet

    for inlet in watershed.inlets: 
        i = 0
        while i < len(rows) - 1:
            for subbasin in rows[i]:
                if subbasin == inlet:
                    
                    # find the position of the subbasin in the chart

                    j = rows[i].index(inlet)

                    if i > 0:

                        # figure out where the subbasins point
                        
                        updowns = [watershed.updown[s] for s in rows[i-1]]
                            
                        # if first or last, add it there in the row above

                        if   j == 0:                rows[i-1].insert(0, 'inlet')
                        elif j == len(rows[i]) - 1: rows[i-1].append('inlet')
                        else:

                            # find the place to add in the preceeding row 

                            n = updowns.index(rows[i][j-1]) + 1
                            rows[i-1].insert(n, 'inlet')
            i += 1

    # write the subbasin boxes to the chart


    middle = math.ceil(w // (rwidth + xgap)) // 2
    last = 0
    for i in range(len(rows)):

        row = rows[i]
        
        y = (ygap + rheight) * i + theight

        # figure out which cell to put in the main column

        if i == 0:
            main = row[(len(row) - 1) // 2]
        #elif i == 1 and len(watershed.inlets) > 0: 
        #    main = row[(len(row) - 1) // 2]
        elif i < len(rows) - 1:
            main = watershed.updown[rows[i-1][last]]
        else: main = 'outlet'

        start = middle - row.index(main)

        if i < len(rows) - 1: next_row = rows[i + 1]

        for subbasin in row:
            x = (rwidth + xgap) * (start + row.index(subbasin))
            r = patches.Rectangle((x, y), rwidth, rheight, fill = False)
            if subbasin != 'outlet': ax.add_patch(r)

            b = ax.text(x + rwidth / 2, y + rheight / 2, subbasin,
                        horizontalalignment = 'center',
                        verticalalignment   = 'center')

            # draw the arrow

            if i < len(rows) - 1:

                x1 = x + rwidth / 2

                if i == 0: 
                    x2 = x1
                elif i < len(rows) - 2 and subbasin != 'inlet':
                    next = watershed.updown[subbasin]
                    next_start = middle - next_row.index(watershed.updown[main])
                    x2 = ((rwidth + xgap) * (next_start + next_row.index(next))
                          + rwidth / 2)
                elif subbasin == 'inlet':
                    next = watershed.inlets[0]
                    next_start = middle - next_row.index(watershed.updown[main])
                    x2 = ((rwidth + xgap) * (next_start + next_row.index(next))
                          + rwidth / 2)
                else:
                    next_start = middle
                    x2 = ((rwidth + xgap) * (middle) + rwidth / 2)

                a = pyplot.arrow(x1, y + rheight, x2 - x1, ygap, 
                              head_width = awidth, head_length = aheight, 
                              fc = 'k', ec = 'k', length_includes_head = True)
                ax.add_patch(a)


        last = row.index(main)
        i += 1

    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(output, dpi = 500)

    pyplot.close()

def plot_NCDC(HUC8, sfile, bfile, pfile = None, efile = None, tfile = None, 
              snowfile = None, centroids = True, radius = None, 
              patchcolor = None, output = None, show = False, verbose = True):
    """Makes a plot of all the hourly precipitation stations of a watershed
    defined by "bfile" with subbasin defined by "sfile" from the source 
    precipitation shapefile "pfile"."""

    if output is None: 
        output = os.getcwd() + '/%sprecipitation' % HUC8

    if verbose: 
        print('generating plot of watershed %s NCDC stations\n' % HUC8)

    fig = pyplot.figure()
    subplot = fig.add_subplot(111, aspect = 'equal')
    subplot.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # add the title

    description = 'NCDC Meteorological Stations'
    title = 'Cataloging Unit %s\n%s' % (HUC8, description)
    subplot.set_title(title, fontsize = 14)

    # open up and show the catchments

    if patchcolor is None: facecolor = (1,0,0,0.)
    else:                  facecolor = patchcolor

    b = Reader(bfile, shapeType = 5)

    points = np.array(b.shape(0).points)
    subplot.add_patch(make_patch(points, facecolor = facecolor, width = 1.))

    extent = get_boundaries(b.shapes(), space = 0.02)

    xmin, ymin, xmax, ymax = extent

    # add the subbasin file

    s = Reader(sfile, shapeType = 5)

    # make patches of the subbasins

    for i in range(len(s.records())):
        shape = s.shape(i)
        points = np.array(shape.points)
        subplot.add_patch(make_patch(points, facecolor, width = 0.15))

    plots = [] # keep track of the scatterplots
    names = [] # keep track of names for the legend
        
    # add the subbasin centroids

    if centroids:

        cx_index = s.fields.index(['CenX', 'N', 12, 6]) - 1
        cy_index = s.fields.index(['CenY', 'N', 12, 6]) - 1

        centroids = [[r[cx_index], r[cy_index]] for r in s.records()]
        xs, ys = zip(*centroids)
        cplot = subplot.scatter(xs, ys, marker = '+', c = 'pink', s = 15)
        plots.append(cplot)
        names.append('Centroids')

    # add a circle showing around subbasin "radius" showing the gages within
    # the radius for a given subbasin

    if radius is not None:

        comid_index = s.fields.index(['ComID',    'N',  9, 0]) - 1
        cx_index    = s.fields.index(['CenX',     'N', 12, 6]) - 1
        cy_index    = s.fields.index(['CenY',     'N', 12, 6]) - 1
        area_index  = s.fields.index(['AreaSqKm', 'N', 10, 2]) - 1

        comids = ['{}'.format(r[comid_index]) for r in s.records()]
        cxs    = [r[cx_index]    for r in s.records()]
        cys    = [r[cy_index]    for r in s.records()]
        areas  = [r[area_index]  for r in s.records()]

        try: i = comids.index(radius)
        except: i = 0
        
        c = [cxs[i], cys[i]]

        radii = [math.sqrt(a / math.pi) for a in areas]

        # scale kms to degrees

        km  = get_distance([xmin, ymin], [xmax, ymax])
        deg = math.sqrt((xmin - xmax)**2 + (ymax - ymin)**2)

        r = sum(radii) / len(radii) * deg / km * 5

        circle = pyplot.Circle(c, radius = r, edgecolor = 'black', 
                               facecolor = 'yellow', alpha = 0.5) 
        subplot.add_patch(circle)
        subplot.scatter(c[0], c[1], marker = '+', c = 'black')

    # add the precipitation gage points

    if pfile is not None:

        f = Reader(pfile, shapeType = 1)

        gage_shapes = f.shapes()
        gage_records = f.records()
        precip_index = f.fields.index(['avg_precip', 'N', 6, 2]) - 1
        precips = [r[precip_index] for r in gage_records]
        gage_points = [g.points[0] for g in gage_shapes]
        x1, y1 = zip(*gage_points)
        gplot = subplot.scatter(x1, y1, s = precips, marker = 'o', c = 'b')
        plots.append(gplot)
        names.append('Hourly Precipitation\nGages') 

    # add the pan evaporation points

    if efile is not None:

        f = Reader(efile, shapeType = 1)

        evap_shapes = f.shapes()
        evap_records = f.records()
        evap_index = f.fields.index(['av_evap_mm', 'N', 6,  2]) - 1
        evaps = [r[evap_index] / 25.4 for r in evap_records]
        evap_points = [e.points[0] for e in evap_shapes]
        x2, y2 = zip(*evap_points)
        eplot = subplot.scatter(x2, y2, s = evaps, marker = 'o', c = 'g')
        plots.append(eplot)
        names.append('Pan Evaporation\nStations')

    # add the temperature station points

    if tfile is not None:

        f = Reader(tfile, shapeType = 1)

        temp_shapes = f.shapes()
        temp_records = f.records()
        temp_points = [t.points[0] for t in temp_shapes]
        x2, y2 = zip(*temp_points)
        tplot = subplot.scatter(x2, y2, marker = 's', c = 'r')
        plots.append(tplot)
        names.append('Temperature\nStations')

    # add the snowdepth station points

    if snowfile is not None:

        f = Reader(snowfile, shapeType = 1)

        snow_shapes = f.shapes()
        snow_records = f.records()
        snow_points = [s.points[0] for s in snow_shapes]
        x2, y2 = zip(*snow_points)
        snowplot = subplot.scatter(x2, y2, marker = 'o', c = 'gray', 
                                   alpha = 0.5)
        plots.append(snowplot)
        names.append('Snowdepth\nStations')

    # add a legend

    #leg = subplot.legend(plots, names)

    leg = subplot.legend(plots, names, bbox_to_anchor = (1.0, 0.5), 
                         loc = 'center left')
    #legtext = leg.get_texts()
    #pyplot.setp(legtext, fontsize = 10)

    #subplot.set_position([0.125, 0.1, 0.6, 0.8])

    # set the labels

    subplot.set_xlabel('Longitude, Decimal Degrees', size = 13)
    subplot.set_ylabel('Latitude, Decimal Degrees',  size = 13)

    #subplot.set_xlim([-93.85, -91.05])
    #subplot.set_ylim([40.7, 42.2])

    # show it

    pyplot.savefig(output)
    if show: pyplot.show()

    pyplot.clf()
    pyplot.close()

def plot_ghcnd(station, start = None, end = None, show = False, output = None,
               verbose = True):
    """Makes a plot of the data from a GHCND station."""

    # some error handling

    precip_data = [(t, p) for t, p in station.precip 
                   if p >= 0 and start <= t and t <= end]
    if len(precip_data) == 0: return
    else: times, precip = zip(*precip_data)

    try:
        if start is None: start = station.precip[0][0]
        if end is None:   end   = station.precip[-1][0]
    except:
        print('warning: no data present')
        return

    if verbose: print('making a plot for {}'.format(station.name))

    # make the plot

    fig, subs = pyplot.subplots(5, 1, sharex = True, figsize = (8, 10))

    vars = station.name, start, end
    fig.suptitle('{} Climate Data {:%m-%d-%Y} to {:%m-%d-%Y}'.format(*vars), 
                 size = 14)

    i = 0

    subs[i].fill_between(times, precip, color = 'cyan',
                         label = 'precipitation')
    subs[i].set_ylabel('Precipitation (mm)', color = 'cyan')

    i = 1

    tmaxdata = [(t, T) for t, T in station.tmax
                if -50 < T and start <= t and t <= end]
    if len(tmaxdata) > 0: times, temp = zip(*tmaxdata)
    else:
        times = [start, start + datetime.timedelta(days = 1), 
                 end - datetime.timedelta(days = 1), end]
        temp = [0, None, None, 1]

    subs[i].plot_date(times, temp, fmt = '-', color = 'red', lw = 0.5, 
                      label = 'max temperature')

    tmindata = [(t, T) for t, T in station.tmin
                if -50 < T and start <= t and t <= end ]
    if len(tmindata) > 0: times, temp = zip(*tmindata)
    else:
        times = [start, start + datetime.timedelta(days = 1), 
                 end - datetime.timedelta(days = 1), end]
        temp = [0, None, None, 1]

    subs[i].plot_date(times, temp, fmt = '-', color = 'blue', lw = 0.5, 
                      label = 'min temperature')
    subs[i].set_ylabel('Temperature (\u00B0C)', color = 'red')

    i = 2
    
    winddata = [(t, w) for t, w in station.wind
                if 0 <= w and start <= t and t <= end]
    if len(winddata) > 0: times, wind = zip(*winddata)
    else:
        times = [start, start + datetime.timedelta(days = 1), 
                 end - datetime.timedelta(days = 1), end]
        wind = [0, None, None, 1]

    subs[i].plot_date(times, wind, fmt = '-', color = 'pink', lw = 0.5, 
                      label = 'wind')
    subs[i].set_ylabel('Wind Speed (m/s)', color = 'pink')

    i = 3
    
    snowdata = [(t, s) for t, s in station.snowdepth
                if 0 <= s and start <= t and t <= end]
    if len(snowdata) > 0:
        times, snow = zip(*snowdata)
        subs[i].fill_between(times, snow, color = 'gray',
                             alpha = 0.5, label = 'snowdepth')
    else:
        times = [start, start + datetime.timedelta(days = 1), 
                 end - datetime.timedelta(days = 1), end]
        snow = [0, None, None, 1]
        subs[i].plot_date(times, snow, color = 'gray', lw = 0.5, 
                          label = 'snowdepth')

    subs[i].set_ylabel('Snow Depth (mm)', color = 'gray')

    i = 4

    evapdata = [(t, e) for t, e in station.evap
                if 0 <= e and start <= t and t <= end]
    if len(evapdata) > 0:
        times, evap = zip(*evapdata)
        subs[i].fill_between(times, evap, color = 'green',
                             alpha = 0.5, label = 'evaporation')
    else:
        times = [start, start + datetime.timedelta(days = 1), 
                 end - datetime.timedelta(days = 1), end]
        evap = [0, None, None, 1]
        subs[i].plot_date(times, evap, label = 'evaporation')

    subs[i].set_ylabel('Pan Evaporation (mm)', color = 'green')

    subs[-1].set_xlabel('Date', size = 12)
    subs[0].xaxis.set_major_locator(MaxNLocator(10))

    if output is not None: pyplot.savefig(output)

    if show: pyplot.show()

def plot_precipitation(precip_file, HUC8, year, tstep = 'hourly', 
                       n = 8, output = None):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if output is None: 
        output = os.getcwd() + '/%d%sprecipitation' % (year, tstep)
    with open(precip_file, 'rb') as f: precipitation = pickle.load(f)

    if len(precipitation) < n: n = len(precipitation) - 1
    if n % 2 != 0: n += -1

    # set the dates
    
    start_date = datetime.datetime(year, 1, 1)
    end_date   = datetime.datetime(year + 1, 1, 1)

    # set the cutoff limits for the number of stations

    totals = [precipitation[p].get_rainfall() for p in precipitation]

    totals.sort()
    totals.reverse()
    cutoff = totals[n - 1]

    # make the figure and subplots

    fig, subs = pyplot.subplots(n, 1, sharex = True, sharey = True,
                                figsize = [8.5, 11])

    fig.suptitle('%d Precipitation Data, HUC %s' % (year, HUC8), size = 14)

    i = 0
    for p in precipitation:

        if (precipitation[p].get_rainfall() >= cutoff):

            hourly = precipitation[p].make_timeseries(start_date, end_date)

            if hourly is None: # no data available, fill with zeros
                hourly = [0. for i in 
                          range(int((end_date-start_date).total_seconds() / 
                                    3600))]
            else:              # convert hourly hundredths of inch to inches

                hourly = [h / 100 for h in hourly]

            if tstep == 'hourly':
                dates = [start_date + i * datetime.timedelta(hours = 1) 
                         for i in range(len(hourly))]
                ts = hourly
                l = 'hr'
                
            elif tstep == 'daily':
                daily = aggregate_hourly_daily(hourly)
                dates = [start_date + i * datetime.timedelta(hours = 24) 
                         for i in range(len(daily))]
                ts = daily
                l = 'd'

            subs[i].plot_date(x = dates, y = ts, fmt = 'r-')

            subs[i].set_ylabel('Prec (in)')
            subs[i].yaxis.set_major_locator(MaxNLocator(4))
            subs[i].text(0.02, 0.95, precipitation[p].name + ' %d' % p, 
                         transform = subs[i].transAxes, ha = 'left', va = 'top',
                         size = 8)
            subs[i].text(0.98, 0.95, 'Total = %.1f in' % sum(ts), transform = 
                         subs[i].transAxes, ha = 'right', va = 'top', size = 8)
            i+=1

    fig.autofmt_xdate()
    subs[-1].set_xlabel('Date')
    for tick in subs[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    #pyplot.show()
    pyplot.savefig(output)
    pyplot.clf()
    pyplot.close()

def plot_evaporation(evap_file, HUC8, year, output = None):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if output is None: 
        output = os.getcwd() + '/%devaporation' % year
    with open(evap_file, 'rb') as f: evaporation = pickle.load(f)

    # set the dates
    
    start_date = datetime.datetime(year, 1, 1, 0, 0)
    end_date   = datetime.datetime(year + 1, 1, 1, 0, 0)

    # make the figure and subplots

    fig, subs = pyplot.subplots(len(evaporation), 1, sharex = True, 
                                sharey = True, figsize = [8.5, 11])

    fig.suptitle('%d Evaporation Data, HUC %s' % (year, HUC8), size = 14)

    if len(evaporation) == 1: subs = [subs]

    i = 0
    for e in evaporation:

        daily = evaporation[e].make_timeseries(start_date, end_date)

        if daily is not None:
            tot = 0
            for d in daily:
                if d is not None: tot += d / 10
            tot = round(tot)
        else: tot = 0

        # convert NoneTypes to zeros and convert to mm from tenths of mm

        ts = []
        for d in daily:
            if d is None: ts.append(0)
            else: ts.append(d / 10)

        if daily is None:  # no data available, use zeros
            ts = [0. for i in 
                  range(int((end_date-start_date).total_seconds() / 
                            3600 / 24))]

        dates = [start_date + i * datetime.timedelta(days = 1) 
                 for i in range(len(ts))]

        subs[i].plot_date(x = dates, y = ts, fmt = 'r-')

        subs[i].set_ylabel('Evap (mm)')
        subs[i].yaxis.set_major_locator(MaxNLocator(4))
        subs[i].text(0.02, 0.95, evaporation[e].name + ' %d' % e, 
                     transform = subs[i].transAxes, ha = 'left', 
                     va = 'top', size = 8)
        subs[i].text(0.98, 0.95, 'Total = %d mm' % tot, transform
                     = subs[i].transAxes, ha = 'right', va = 'top', 
                     size = 8)
        i+=1

    subs[-1].set_xlabel('Date')
    for tick in subs[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    fig.autofmt_xdate()
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    #pyplot.show()
    pyplot.savefig(output)
    pyplot.clf()
    pyplot.close()

def plot_hourly_evap(evap_file, HUC8, start_date, end_date, output = None):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if output is None: 
        output = os.getcwd() + '/%devaporation'
    with open(evap_file, 'rb') as f: evaporation = pickle.load(f)

    # make the figure and subplots

    fig, subs = pyplot.subplots(len(evaporation) + 1, 1, sharex = True, 
                                sharey = True, figsize = [8.5, 11])

    fig.suptitle('Evaporation Disaggregation HUC %s (%s to %s)' % 
                 (HUC8, start_date.date(), end_date.date()), size = 14)

    i = 0
    for e in evaporation:

        lat, long = evaporation[e].latitude, evaporation[e].longitude

        daily = evaporation[e].make_timeseries(start_date, end_date)
        
        hourly = hourly_evaporation(daily, start_date, lat, long)

        if hourly is None:
            hourly = [None for i in 
                      range(int((end_date-start_date).total_seconds() / 
                                3600))]
            tot = 0
        else: tot = sum(hourly)
        
        dates = [start_date + i * datetime.timedelta(hours = 1) 
                 for i in range(len(hourly))]

        subs[i].plot_date(x = dates, y = hourly, fmt = 'r-')
        subs[i].set_xlim([start_date, end_date])
        subs[i].set_ylabel('Evap (mm)')
        subs[i].yaxis.set_major_locator(MaxNLocator(4))
        subs[i].text(0.02, 0.95, evaporation[e].name + ' %d' % e, transform = 
                     subs[i].transAxes, ha = 'left', va = 'top', size = 8)
        subs[i].text(0.98, 0.95, 'Total = %d mm' % tot, transform
                     = subs[i].transAxes, ha = 'right', va = 'top', size = 8)
        i+=1

    watershed = make_evaporation(evaporation, start_date, end_date)
    tot = sum(watershed)
    subs[i].plot_date(x = dates, y = watershed, fmt = 'r-')
    subs[i].set_xlim([start_date, end_date])
    subs[i].set_ylabel('Evap (mm)')
    subs[i].yaxis.set_major_locator(MaxNLocator(4))
    subs[i].text(0.02, 0.95, 'HUC %s' % HUC8, transform = 
                 subs[i].transAxes, ha = 'left', va = 'top', size = 8)
    subs[i].text(0.98, 0.95, 'Total = %d mm' % tot, transform
                 = subs[i].transAxes, ha = 'right', va = 'top', size = 8)

    subs[-1].set_xlabel('Date')
    for tick in subs[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    fig.autofmt_xdate()
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    #pyplot.show()
    pyplot.savefig(output)
    pyplot.close()

def plot_temperature(temp_file, HUC8, year, output = None, verbose = False):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if verbose: print('plotting {} for year {}'.format(HUC8, year))
    if output is None: 
        output = os.getcwd() + '/%dtemperature' % year
    with open(temp_file, 'rb') as f: temperature = pickle.load(f)

    # set the dates
    
    start_date = datetime.datetime(year, 1, 1, 0, 0)
    end_date   = datetime.datetime(year + 1, 1, 1, 0, 0)

    # make the figure and subplots

    fig, subs = pyplot.subplots(len(temperature), 1, sharex = True, 
                                sharey = True, figsize = [8.5, 11])

    fig.suptitle('%d Daily High and Low Temperatures, HUC %s' % (year, HUC8), 
                 size = 14)

    i = 0

    if len(temperature) == 1: subs = [subs]
    for t in temperature:

        tmax, tmin = temperature[t].make_timeseries(start_date, end_date)

        # no data available

        if (tmax is None or tmin is None): 
            tmax  = [0, None, None, 20]
            tmin  = tmax
            dates = [start_date, start_date + datetime.timedelta(days = 1),
                     end_date - datetime.timedelta(days = 1), end_date]

        dates = [start_date + i * datetime.timedelta(days = 1) 
                 for i in range(len(tmax))]

        if (all([tm is None for tm in tmax]) or 
            all([tm is None for tm in tmin])):
            tmax  = [0, None, None, 20]
            tmin  = tmax
            dates = [start_date, start_date + datetime.timedelta(days = 1),
                     end_date - datetime.timedelta(days = 1), end_date]

        subs[i].plot_date(x = dates, y = tmax, fmt = 'r-', label = 'High')
        subs[i].plot_date(x = dates, y = tmin, fmt = 'b-', label = 'Low')

        subs[i].set_ylabel('Temp (\u00B0C)')
        subs[i].yaxis.set_major_locator(MaxNLocator(4))
        subs[i].text(0.02, 0.95, temperature[t].name + ' {}'.format(t), 
                     transform = subs[i].transAxes, ha = 'left', va = 'top',
                     size = 8)
        i+=1

    fig.autofmt_xdate()
    for tick in subs[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
            
    subs[-1].set_xlabel('Date', size = 12)
    subs[-1].legend(bbox_to_anchor = (0.98, 0.02), loc = 'lower right', 
                    borderaxespad = 0.)
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    #pyplot.show()
    pyplot.savefig(output)
    pyplot.clf()
    pyplot.close()

def plot_hourly_temp(temp_file, HUC8, start_date, end_date, output = None):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if output is None:
        output = os.getcwd() + '/%dtemperature' % start_date.year
    with open(temp_file, 'rb') as f: temperature = pickle.load(f)

    # make the figure and subplots

    fig, subs = pyplot.subplots(len(temperature), 1, sharex = True, 
                                sharey = True, figsize = [8.5, 11])

    fig.suptitle('Temperature Disaggregation, HUC %s (%s to %s)' % 
                 (HUC8, start_date.date(), end_date.date()), 
                 size = 14)

    i = 0

    for t in temperature:

        tmax, tmin = temperature[t].make_timeseries(start_date, end_date)
        hourly = hourly_temperature(tmax, tmin)

        if hourly is None:  # no data available, use zeros
            hourly = [None for i in 
                      range(int((end_date-start_date).total_seconds() / 
                                3600))]

        dates = [start_date + i * datetime.timedelta(hours = 1) 
                 for i in range(len(hourly))]

        subs[i].plot_date(x = dates, y = hourly, fmt = 'g-', label = 'Hourly')

        dates = [start_date + i * datetime.timedelta(days = 1) 
                 for i in range(len(tmax))]

        subs[i].plot_date(x = dates, y = tmin, fmt = 'b-', label = 'Tmin')
        subs[i].plot_date(x = dates, y = tmax, fmt = 'r-', label = 'Tmax')
        subs[i].set_ylabel('Temp (' + '\u00B0' + 'C)')
        subs[i].yaxis.set_major_locator(MaxNLocator(4))
        subs[i].text(0.02, 0.95, temperature[t].name + ' %d' % t, 
                     transform = subs[i].transAxes, ha = 'left', va = 'top',
                     size = 8)
        subs[i].legend(bbox_to_anchor = (0.98, 0.02), loc = 'lower right', 
                       borderaxespad = 0., fontsize = 9)
        i+=1

    fig.autofmt_xdate()
    for tick in subs[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    subs[-1].set_xlabel('Date', size = 12)
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    pyplot.show()
    #pyplot.savefig(output)
    #pyplot.close()

def plot_hourly(evaporation, temperature, precipitation, HUC8, start_date, 
                output = None):
    """Generates a series of subplots of the time series of precipitation
    data for a watershed."""

    if output is None: output = os.getcwd() + '/%dhourly'

    # make the figure and subplots

    fig, subs = pyplot.subplots(3, 1, sharex = True, 
                                sharey = False, figsize = [8.5, 11])

    dates = [start_date + i * datetime.timedelta(hours = 1)
             for i in range(len(evaporation))]

    fig.suptitle('Hourly Disaggregation HUC %s (%s to %s)' % 
                 (HUC8, start_date.date(), dates[-1].date()), size = 14)

    tot = sum(evaporation)

    subs[0].plot_date(x = dates, y = evaporation, fmt = 'r-')
    subs[0].set_xlim([dates[0], dates[-1]])
    subs[0].set_ylabel('Evaporation (mm)')
    subs[0].yaxis.set_major_locator(MaxNLocator(4))
    subs[0].text(0.02, 0.95, 'HUC %s' % HUC8, transform = 
                 subs[0].transAxes, ha = 'left', va = 'top', size = 8)
    subs[0].text(0.98, 0.95, 'Total = %d mm' % tot, transform
                 = subs[0].transAxes, ha = 'right', va = 'top', size = 8)

    subs[1].plot_date(x = dates, y = temperature, fmt = 'r-')
    subs[1].set_xlim([dates[0], dates[-1]])
    subs[1].set_ylabel('Temperature (C)')
    subs[1].yaxis.set_major_locator(MaxNLocator(4))
    subs[1].text(0.02, 0.95, 'HUC %s' % HUC8, transform = 
                 subs[1].transAxes, ha = 'left', va = 'top', size = 8)

    subs[2].plot_date(x = dates, y = precipitation, fmt = 'r-')
    subs[2].set_xlim([dates[0], dates[-1]])
    subs[2].set_ylabel('Precipitation (mm)')
    subs[2].yaxis.set_major_locator(MaxNLocator(4))
    subs[2].yaxis.set_major_locator(MaxNLocator(4))
    subs[2].text(0.02, 0.95, 'HUC %s' % HUC8, transform = 
                 subs[2].transAxes, ha = 'left', va = 'top', size = 8)
    subs[2].text(0.98, 0.95, 'Total = %d mm' % sum(precipitation), transform = 
                 subs[2].transAxes, ha = 'right', va = 'top', size = 8)
 
    subs[2].set_xlabel('Date')
    for tick in subs[2].xaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    fig.autofmt_xdate()
    pyplot.subplots_adjust(top = 0.95, bottom = 0.1)
    pyplot.show()
    #pyplot.savefig(output)
    #pyplot.close()


