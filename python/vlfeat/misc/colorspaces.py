import numpy

def vl_xyz2lab(I,il='E'):
# VL_XYZ2LAB  Convert XYZ color space to LAB
#   J = VL_XYZ2LAB(I) converts the image from XYZ format to LAB format.
#
#   VL_XYZ2LAB(I,IL) uses one of the illuminants A, B, C, E, D50, D55,
#   D65, D75, D93. The default illuminant is E.
#
#   See also:: VL_XYZ2LUV(), VL_HELP().

# AUTORIGHTS
# Copyright (C) 2007-10 Andrea Vedaldi and Brian Fulkerson
#
# This file is part of VLFeat, available under the terms of the
# GNU GPLv2, or (at your option) any later version.

    def f(a):
        k = 903.3  
        b=numpy.zeros(a.shape) 
        b[a>0.00856] = a[a>0.00856]**(1/3.) 
        b[a<=0.00856] = (k*a[a<=0.00856] + 16)/116. 
        return b

    il=il.lower()

    if il=='a': 
        xw = 0.4476 
        yw = 0.4074 
    elif il=='b':
        xw = 0.3324 
        yw = 0.3474 
    elif il=='c':
        xw = 0.3101 
        yw = 0.3162 
    elif il=='e':
        xw = 1./3 
        yw = 1./3
    elif il=='d50':
        xw = 0.3457 
        yw = 0.3585 
    elif il=='d55':
        xw = 0.3324 
        yw = 0.3474 
    elif il=='d65':
        xw = 0.312713 
        yw = 0.329016 
    elif il=='d75':
        xw = 0.299 
        yw = 0.3149 
    elif il=='d93':
        xw = 0.2848
        yw = 0.2932 

    J=numpy.zeros(I.shape) 

# Reference white
    Yw = 1.0 
    Xw = xw/yw 
    Zw = (1-xw-yw)/yw * Yw

# XYZ components
    X = I[:,:,0] 
    Y = I[:,:,1] 
    Z = I[:,:,2] 

    x = X/Xw 
    y = Y/Yw 
    z = Z/Zw 

    L = 116 * f(y) - 16 
    a = 500*(f(x) - f(y))
    b = 200*(f(y) - f(z)) 
    J = numpy.rollaxis(numpy.array([L,a,b]),0,3)
    return J

def vl_rgb2xyz(I,workspace="CIE"):
     #VL_RGB2XYZ  Convert RGB color space to XYZ
       #J=VL_RGB2XYZ(I) converts the CIE RGB image I to the image J in 
       #CIE XYZ format. CIE RGB has a white point of R=G=B=1.0
    
       #VL_RGB2XYZ(I,WS) uses the specified RGB working space WS. The
       #function supports the following RGB working spaces:
    
       #* `CIE'    E illuminant, gamma=2.2
       #* `Adobe'  D65 illuminant, gamma=2.2
    
       #The default workspace is CIE.
    
       #See also:: VL_XYZ2RGB(), VL_HELP().

     #AUTORIGHTS
     #Copyright (C) 2007-10 Andrea Vedaldi and Brian Fulkerson
    
     #This file is part of VLFeat, available under the terms of the
     #GNU GPLv2, or (at your option) any later version.

    M,N,K = I.shape

    if not K==3:
        print('I must be a MxNx3 array.') 
        exit(0)

    #I=im2double(I) ;

    if workspace=='CIE':
         #CIE: E illuminant and 2.2 gamma
        A = numpy.array([
          [0.488718,   0.176204,   0.000000],
          [0.310680,   0.812985,   0.0102048],
          [0.200602,   0.0108109, 0.989795 ]]).T
        gamma = 2.2 

    if workspace=='Adobe':
         #Adobe 1998: D65 illuminant and 2.2 gamma
        A = numpy.array([
          [0.576700,   0.297361,   0.0270328],
          [0.185556,   0.627355,   0.0706879],
          [0.188212,   0.0752847,  0.99124 ]]).T
        gamma = 2.2 

    I = (I**gamma).reshape(M*N, K) ;
    J = numpy.dot(A,I.T) 
    J = J.T.reshape(M, N, K) 
    return J
