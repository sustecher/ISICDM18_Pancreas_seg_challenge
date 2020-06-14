function h = fspecial3(type,Hsize,sigma)
%FSPECIAL3 Create predefined 3-D filters.
%   H = FSPECIAL3(TYPE) creates a three-dimensional filter H of the
%   specified type. Possible values for TYPE are:
%
%     'average'   averaging filter
%     'ellipsoid' ellipsoidal averaging filter
%     'gaussian'  Gaussian lowpass filter
%     'laplacian' filter approximating the 3-D Laplacian operator
%     'log'       Laplacian of Gaussian filter
%     'prewitt'   Prewitt horizontal edge-emphasizing filter
%     'sobel'     Sobel horizontal edge-emphasizing filter
%
%   Depending on TYPE which can be a string or a char vector, FSPECIAL3 may
%   take additional parameters which you can supply.  These parameters all
%   have default values.
%
%   H = FSPECIAL3('average',HSIZE) returns an averaging filter H of size
%   HSIZE. HSIZE can be a 3-element vector specifying the number of rows,
%   columns and planes in H or a scalar, in which case H is a cubic array.
%   The default HSIZE is [5 5 5]. See Attached Note.
%
%   H = FSPECIAL3('ellipsoid',SEMIAXES) returns an ellipsoidal averaging
%   filter with SEMIAXES defining length of the principal semi-axes of the
%   ellipsoid. SEMIAXES can be a 3-element vector specifying the length of
%   the prinicipal semi-axes in rows, columns and planes in H or it can be
%   a scalar, in which case H is a sphere. The filter H lies within a
%   cuboidal array of size 2*ceil(SEMIAXES)+1.The default value of SEMIAXES is 5.
%
%   H = FSPECIAL3('gaussian',HSIZE,SIGMA) returns a Gaussian lowpass filter
%   of size HSIZE with standard deviation SIGMA (positive). HSIZE can be a
%   3-element vector specifying the number of rows, columns and planes in H
%   or a scalar, in which case H is a cubic array. SIGMA can be a 3-element
%   vector with positive values or a scalar. If sigma is a scalar, a cubic
%   Gaussian kernel is used. The default HSIZE is [5 5 5], the default
%   SIGMA is 1. See Attached Note.
%
%   H = FSPECIAL3('laplacian',GAMMA1,GAMMA2) returns a 3-by-3-by-3 filter
%   approximating the shape of the three-dimensional Laplacian operator.
%   The parameters GAMMA1 and GAMMA2 control the shape of the Laplacian and
%   must be in the range 0.0 to 1.0 and the sum of these values must not
%   exceed 1.0 either. Default values of GAMMA1 and GAMMA2 is 0.
%
%   H = FSPECIAL3('log',HSIZE,SIGMA) returns Laplacian of Gaussian filter of
%   size HSIZE with standard deviation SIGMA (positive). HSIZE can be a
%   3-element vector specifying the number of rows, columns and planes in H
%   or a scalar, in which case H is a cubic array. SIGMA can be a 3-element
%   vector with positive values or a scalar. If sigma is a scalar, a cubic
%   Gaussian kernel is used.The default HSIZE is [5 5 5], the default SIGMA
%   is 1.
%
%   H = FSPECIAL3('prewitt','direction') returns 3-by-3-by-3 filter that
%   emphasizes gradients in the axis specified by 'direction'.
%   'Direction' can be a string or a char and the possible values it can
%   take are 'X','Y' or 'Z'. The default value of 'direction' is 'X'.
%
%   H = FSPECIAL3('sobel','direction') returns 3-by-3-by-3 filter that
%   emphasizes gradients in the axis specified by 'direction' utilizing the
%   smoothing effect in the other directons. 'Direction' can be a string or
%   a char and the possible values it can take are 'X','Y' or 'Z'. The
%   default value of 'direction' is 'X'.
%
%   Class Support
%   -------------
%   H is of class double.
%
%   Notes
%   -----
%   For 'gaussian' and 'log' kernels, When a value is provided for sigma
%   and HSIZE is specified as [], a default HSIZE of dimension
%   2*ceil(2*sigma)+1 is chosen.
%   
%   For performing purely average filtering, using imboxfilt3 is the
%   preferred syntax.
%
%   For performing purely gaussian filtering, using imgaussfilt3 is the
%   preferred syntax.
%   
%   The pricipal semi-axes in ellipsoidal filter correspond to the
%   cartesian coordinate system. The length in rows, columns and planes
%   correspond to length in Y,X and Z axes respectively.
%
%   Example 1
%   ---------
%   Smooth an MRI volume with a 3-D ellipsoidal filter.
%
%   load mristack;
%   H = fspecial3('ellipsoid',[7 7 3]);
%   volSmooth = imfilter(mristack,H,'replicate');
%   volshow(volSmooth);
%
%   Example 2
%   ---------
%   Find horizontal edges (vertical gradients) using 3D sobel
%   operator in an MRI volume.
%
%   load mristack;
%   H = fspecial3('sobel','Y');
%   edgesHor = imfilter(mristack,H,'replicate');
%   volshow(edgesHor);
%
%   References
%   ----------
%   3D Laplacian:    
%   Lindeberg, T., Scale-Space Theory in Computer Vision, Kluwer Academic
%   Publishers, 1994 
%   Ter, Haar Romeny Bart M. Geometry-Driven Diffusion in
%   Computer Vision. Kluwer Academic Publishers, 1994.
%
%   3D sobel:
%   K. Engel (2006). Real-time volume graphics,. pp. 112?114.
%
%   See also IMFILTER, IMBOXFILT3, IMGAUSSFILT3, EDGE3, PERMUTE, FLIPUD

%   Copyright 2018 The MathWorks, Inc.


switch type
    case 'average' % Smoothing filter
        h   = ones(Hsize)/prod(Hsize);
    case 'ellipsoid'
        % Variable Size refers to the length of SEMI-AXES values
        xr = Hsize(2);
        yr = Hsize(1);
        zr = Hsize(3); 
        % Get the 3D array dimensions = 2*ceil(SEMIAXES)+1
        Hsize = ceil(Hsize);
        xs = Hsize(2);
        ys = Hsize(1);
        zs = Hsize(3);
        [X,Y,Z] = meshgrid(-xs:xs,-ys:ys,-zs:zs);
        h = (1 - X.^2/xr^2 - Y.^2/yr^2 -Z.^2/zr^2) >= 0;
        h = h/sum(h(:));
    case 'gaussian'
        logFlag=false;
        h = gaussianAlgoHelper(Hsize,sigma,logFlag);
        
    case 'log'
        logFlag=true;
        h = gaussianAlgoHelper(Hsize,sigma,logFlag);
  
    case 'laplacian'
        h1(:,:,1) = [0 0 0; 0 1 0; 0 0 0];
        h1(:,:,2) = [0 1 0; 1 -6 1; 0 1 0];
        h1(:,:,3) = h1(:,:,1);
        
        h2(:,:,1) = [0 1 0; 1 0 1; 0 1 0];
        h2(:,:,2) = [1 0 1; 0 -12 0; 1 0 1];
        h2(:,:,3) = h2(:,:,1);
        h2 = 0.25*h2;
        
        h3(:,:,1) = [1 0 1; 0 0 0; 1 0 1];
        h3(:,:,2) = [0 0 0; 0 -8 0; 0 0 0];
        h3(:,:,3) = h3(:,:,1);
        h3 = 0.25*h3;
        
        h = (1 - gamma(1) - gamma(2))*h1 + gamma(1)*h2 + gamma(2)*h3;
        
    case 'prewitt'
        ht =[ 1 0 -1 ; 1 0 -1 ; 1 0 -1 ];
        h = cat(3,ht,ht,ht);
        if strcmp(direction,'Y')
            h = permute(h,[2 1 3]);
        elseif strcmp(direction,'Z')
            h = permute(h,[3 1 2]);
        end
    case 'sobel'
        h(:,:,1) =[1 0 -1; 2 0 -2 ; 1 0 -1];
        h(:,:,2) =2*h(:,:,1);
        h(:,:,3) = h(:,:,1);
        if strcmp(direction,'Y')
            h = permute(h,[2 1 3]);
        elseif strcmp(direction,'Z')
            h = permute(h,[3 1 2]);
        end
end

function [type, Hsize, sigma, gamma, direction ] = ParseInputs(varargin)

% default values
Hsize   = [5 5 5];
sigma  = [1 1 1];
gamma  = [0 0];
direction = 'X';
% Check the number of input arguments.
narginchk(1,3);

% Determine filter type from the user supplied string.
type = varargin{1};
type = validatestring(type,{'gaussian','sobel','prewitt','laplacian','log',...
                    'average','ellipsoid'},mfilename,'TYPE',1);

switch nargin
    case 1
        % FSPECIAL3('average')
        % FSPECIAL3('ellipsoid')
        % FSPECIAL3('gaussian')
        % FSPECIAL3('log')
        % FSPECIAL3('laplacian')
        % FSPECIAL3('prewitt')
        % FSPECIAL3('sobel')
        % Nothing to do here; the default values have
        % already been assigned.        
        
    case 2
       % FSPECIAL3('average',HSIZE)
       % FSPECIAL3('ellipsoid',SEMIAXES)
       % FSPECIAL3('gaussian',HSIZE)
       % FSPECIAL3('log',HSIZE)
       % FSPECIAL3('laplacian',GAMMA1) GAMMA2=0
       % FSPECIAL3('prewitt','direction')
       % FSPECIAL3('sobel','direction')
       Hsize = varargin{2};
       
       switch type
           case {'sobel','prewitt'}
               direction = Hsize;
               direction = validatestring(direction,{'X','Y','Z'},mfilename,'DIRECTION',2);
               
           case 'laplacian'
               validateattributes(Hsize,{'double'},{'nonnegative','real',...
                   'nonempty','finite','scalar','<=',1},...
                   mfilename,'GAMMA1',2);              
               gamma = [Hsize 0];
           case 'ellipsoid'
               validateattributes(Hsize,{'double'},...
                   {'positive','finite','real','nonempty','vector'},...
                   mfilename,'SEMIAXES',2);
               Hsize = sizeTest(Hsize);
           case {'gaussian','log','average'}
               validateattributes(Hsize,{'double'},...
                   {'positive','finite','real','nonempty','integer'},...
                   mfilename,'HSIZE',2);
               Hsize = sizeTest(Hsize);
       end
       
    case 3
        % FSPECIAL3('gaussian',HSIZE,SIGMA) or FSPECIAL3('gaussian',[],SIGMA)
        % FSPECIAL3('log',HSIZE,SIGMA) or FSPECIAL3('log',[],SIGMA)
        % FSPECIAL3('laplacian',GAMMA1,GAMMA2)
        Hsize = varargin{2};
        sigma = varargin{3};
        
        switch type
            case {'gaussian','log'}

                validateattributes(sigma,{'double'},...
                    {'positive','finite','real','nonempty'},...
                    mfilename,'SIGMA',3);
                sigma = sigmaTest(sigma);
                if(isempty(Hsize))
                    Hsize = 2*ceil(2*sigma)+1;
                end
                validateattributes(Hsize,{'double'},...
                    {'positive','finite','real','nonempty','integer'},...
                    mfilename,'HSIZE',2);
                Hsize = sizeTest(Hsize);

            case 'laplacian'
                % GAMMA1 = size, GAMMA2 = sigma
                validateattributes(Hsize,{'double'},{'nonnegative','real',...
                    'nonempty','finite','scalar'},...
                    mfilename,'GAMMA1',2);
                validateattributes(sigma,{'double'},{'nonnegative','real',...
                    'nonempty','finite','scalar'},...
                   mfilename,'GAMMA2',2);
                gamma = [Hsize sigma];
            if (sum(gamma) > 1)
                error(message('images:fspecial3:outOfRangeSumGamma'))
            end
            otherwise
                error(message('images:fspecial3:tooManyArgsForThisFilter'))
        end      
end

function h = gaussianAlgoHelper(Hsize,sigma,logFlag)

xsize = (Hsize(2)-1)/2;
ysize = (Hsize(1)-1)/2;
zsize = (Hsize(3)-1)/2;
[X,Y,Z] = meshgrid(-xsize:xsize,-ysize:ysize,-zsize:zsize);
xsig2 = sigma(2)^2;
ysig2 = sigma(1)^2;
zsig2 = sigma(3)^2;

arg = -(X.^2/(2*xsig2) + Y.^2/(2*ysig2) + Z.^2/(2*zsig2));
h     = exp(arg);
h(h<eps*max(h(:))) = 0;
% Normalize Gaussian: Essentially takes care of the constant
% 1/(2pi*sigma)^0.5 term which only normalizes in continuous domain
sumh = sum(h(:));
if sumh ~= 0
    h  = h/sumh;
end

if logFlag
    % Calculate Laplacian of Gaussian
    h1 = h.*(X.^2/xsig2^2 + Y.^2/ysig2^2 + Z.^2/zsig2^2 - 1/xsig2 -1/ysig2 -1/zsig2);
    h     = h1 - sum(h1(:))/prod(Hsize); % make the filter sum to zero
end
        
        


function Hsize = sizeTest(Hsize)

if numel(Hsize) > 3 || numel(Hsize) == 2
    error(message('images:fspecial3:wrongSizeHSize'))
elseif numel(Hsize)==1
    Hsize = [Hsize Hsize Hsize];
end

function sigma = sigmaTest(sigma)

if numel(sigma) > 3 || numel(sigma) == 2
    error(message('images:fspecial3:wrongSizeSigma'))
elseif numel(sigma)==1
    sigma = [sigma sigma sigma];
end
