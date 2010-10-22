import numpy
import sys
from .. import _vlfeat

from vlfeat.misc.colorspaces import *

eps=1e-10

def vl_flatmap(tree):
    """ Flatten a tree, assigning the label of the root to each node
    [LABELS CLUSTERS] = VL_FLATMAP(MAP) labels each tree of the forest contained
    in MAP. LABELS contains the linear index of the root node in MAP, CLUSTERS
    instead contains a label between 1 and the number of clusters.
    """
    #follow the parents list to the root nodes (where nothing changes)
    shape=list(tree.shape)
    shape.reverse()
    tree=tree.flatten("F")
    i=0 
    while 1:
        i=i+1
        tree_ = tree[tree] 
        if (tree_==tree).all():
            break
        tree= tree_ 
    drop,clusters=numpy.unique(tree, return_inverse=True)
    tree=tree.reshape(shape).T
    clusters=clusters.reshape(shape).T
    return tree,clusters 


def vl_quickshift(image, sigma,maxdist=None,medoid=False):
    """ Perform quickshift segmentation - this function is a thin wrapper and expects an image in LAB space
    Use vl_imseg for easy segmentation.
    """
    N1=image.shape[0]
    N2=image.shape[1]
    K=image.shape[2]
    if maxdist==None:
        maxdist=3*sigma
    q = _vlfeat.vl_quickshift_new(image, N1, N2, K)

    _vlfeat.vl_quickshift_set_kernel_size (q, sigma) 
    _vlfeat.vl_quickshift_set_max_dist     (q, maxdist) 
    _vlfeat.vl_quickshift_set_medoid      (q, medoid) 

    _vlfeat.vl_quickshift_process(q)

    parentsi = _vlfeat.vl_quickshift_get_parents(q)
    parents = parentsi #+ 1
    dists= _vlfeat.vl_quickshift_get_dists(q)
    density= _vlfeat.vl_quickshift_get_density(q)
    #/* Delete quick shift object */
    #_vlfeat.vl_quickshift_delete(q)
    return [parents,dists,density]

def vl_imseg(I,labels):
    """Color an image based on the segmentation
    ISEG = IMSEG(I,LABELS) Labels ISEG with the average color from I of 
    each cluster indicated by LABELS
    """

    [M,N,K] = I.shape
    Q = 0*I
    for k in xrange(0,K):
        acc = numpy.zeros((M,N))
        nrm = numpy.zeros((M,N))
        acc = numpy.bincount(labels.flatten(), weights=I[:,:,k].flatten())
        nrm = numpy.bincount(labels.flatten(),weights=numpy.ones((M,N)).flatten())
        acc = acc / (nrm+eps)
        Q[:,:,k] = acc[labels]

    Q[Q>1]=1
    return Q

def vl_quickseg(
        image,
        ratio,
        kernelsize,
        maxdist):
    """ Compute Quickshift segmentation of image.
    @param image        Float RGB or grayscale image
    @param ratio        Tradeof between color and proximity (between 0 and 1)
    @param kernelsize   Size of distance kernel
    @param maxdist      Maximum distance between modes to be joined into one segment
    """
    if not image.flags['F_CONTIGUOUS']:
        image = numpy.array(image, order='F')		

    # break ties on uniform areas:
    image = image + numpy.random.uniform(0,1,image.shape)/2550

    if image.shape[2] == 1:
      imagex = ratio * image
    else:
      imagex = ratio * vl_xyz2lab(vl_rgb2xyz(image))
      #Ix = Ix(:,:,2:3); % Throw away L

    # Perform quickshift to obtain the segmentation tree, which is already cut by
    # maxdist. If a pixel has no nearest neighbor which increases the density, its
    # parent in the tree is itself, and gaps is inf.
    mapping,gaps,E = vl_quickshift(imagex.copy('F'), kernelsize, maxdist) 

    # Follow the parents of the tree until we have reached the root nodes
    # mapped: a labeled segmentation where the labels are the indicies of the modes
    # in the original image.
    # labels: mapped after having been renumbered 1:nclusters and reshaped into a
    # vector
    mapped, labels = vl_flatmap(mapping) 
    # imseg builds an average description of the region by color
    Iseg=vl_imseg(image, labels)
    return [Iseg, labels,mapping,gaps,E]

def vl_quickvis(I, ratio, kernelsize, maxdist, maxcuts=None):
    """ Create an edge image from a Quickshift segmentation.
    IEDGE = VL_QUICKVIS(I, RATIO, KERNELSIZE, MAXDIST, MAXCUTS) creates an edge
    stability image from a Quickshift segmentation. RATIO controls the tradeoff
    between color consistency and spatial consistency (See VL_QUICKSEG) and
    KERNELSIZE controls the bandwidth of the density estimator (See VL_QUICKSEG,
    VL_QUICKSHIFT). MAXDIST is the maximum distance between neighbors which
    increase the density. 

    VL_QUICKVIS takes at most MAXCUTS thresholds less than MAXDIST, forming at
    most MAXCUTS segmentations. The edges between regions in each of these
    segmentations are labeled in IEDGE, where the label corresponds to the
    largest DIST which preserves the edge. 

    [IEDGE,DISTS] = VL_QUICKVIS(I, RATIO, KERNELSIZE, MAXDIST, MAXCUTS) also
    returns the DIST thresholds that were chosen.

    IEDGE = VL_QUICKVIS(I, RATIO, KERNELSIZE, DISTS) will use the DISTS
    specified

    [IEDGE,DISTS,MAP,GAPS] = VL_QUICKVIS(I, RATIO, KERNELSIZE, MAXDIST, MAXCUTS)
    also returns the MAP and GAPS from VL_QUICKSHIFT.

    See Also: VL_QUICKSHIFT, VL_QUICKSEG
    """
    if not I.flags['F_CONTIGUOUS']:
        I = numpy.array(I, order='F')		

    if maxcuts == None: 
        dists = maxdist;
        maxdist = numpy.max(dists);
        Iseg, labels, mapping, gaps, E = vl_quickseg(I, ratio, kernelsize, maxdist)
    else:
        Iseg, labels, mapping, gaps, E = vl_quickseg(I, ratio, kernelsize, maxdist)
        dists = numpy.unique(numpy.floor(gaps.flatten()))
        dists = dists[1:-1]  # remove the inf thresh and the lowest level thresh
        if len(dists) > maxcuts:
            ind = [int(x) for x in numpy.round(numpy.linspace(0,len(dists)-1, maxcuts))]
            dists = dists[ind]
    Iedge, dists = mapvis(mapping, gaps, dists)
    return [Iedge, dists, mapping, gaps] 


def mapvis(mapping, gaps, maxdist, maxcuts=None):
    """Create an edge image from a Quickshift segmentation.
    IEDGE = MAPVIS(MAP, GAPS, MAXDIST, MAXCUTS) creates an edge
    stability image from a Quickshift segmentation. MAXDIST is the maximum
    distance between neighbors which increase the density. 

    MAPVIS takes at most MAXCUTS thresholds less than MAXDIST, forming at most
    MAXCUTS segmentations. The edges between regions in each of these
    segmentations are labeled in IEDGE, where the label corresponds to the
    largest DIST which preserves the edge. 

    [IEDGE,DISTS] = MAPVIS(MAP, GAPS, MAXDIST, MAXCUTS) also returns the DIST
    thresholds that were chosen.

    IEDGE = MAPVIS(MAP, GAPS, DISTS) will use the DISTS specified

    See Also: VL_QUICKVIS, VL_QUICKSHIFT, VL_QUICKSEG
    """
    if maxcuts == None:
        dists = maxdist
        maxdist = max(dists)
    else:
        dists = unique(floor(gaps[:]))
        dists = dists[1:-1] # remove the inf thresh and the lowest level thresh
        # throw away min region size instead of maxdist?
        dists = dists(dists<maxdist)
        if len(dists) > maxcuts:
            ind = round(linspace(1,len(dists), maxcuts))
            dists = dists(ind)

    Iedge = numpy.zeros(mapping.shape)

    for i in xrange(len(dists)):
        mapdist = mapping.flatten("F")
        s=numpy.where((gaps.flatten("F")>=dists[i]))
        mapdist[s] = s
        mapdist=mapdist.reshape(mapping.shape[1],mapping.shape[0]).T
        mapped,labels = vl_flatmap(mapdist)
        print('%d/%d %d regions\n'%(i, len(dists), len(numpy.unique(mapped))))

        borders = getborders(mapped);
        
        Iedge[borders] = dists[i];
        #Iedge[borders] = Iedge[borders] + 1;
        #Iedge[borders] = i;
    return [Iedge, dists] 

def getborders(mapping):
    """ Get edges within an image by calculating image derivatives
    """
    from scipy.signal import convolve2d
    dx = convolve2d(mapping, numpy.array([[-1, 1]]), 'same')
    dy = convolve2d(mapping, numpy.array([[-1, 1]]).T, 'same')
    borders = (dx != 0)+( dy != 0)
    return borders
