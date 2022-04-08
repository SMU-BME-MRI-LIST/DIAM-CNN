import numpy as np
import tensorflow as tf
PRECISION = 'float32'
EPS = 1E-8

#
# Description:
#Support function for training QSMnet and DIAM-CNN
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

# Data I/O
def load_nii(filename):
    import nibabel as nib
    return nib.load(filename).get_data()


def save_nii(data, filename, filename_sample=''):
    import nibabel as nib
    if filename_sample:
        nib.save(nib.Nifti1Image(data, None, nib.load(filename_sample).header), filename)
    else:
        nib.save(nib.Nifti1Image(data, None, None), filename)


def load_h5(filename, varname='data'):
    import h5py
    with h5py.File(filename, 'r') as f:
        data = f[varname][:]
    return data


def save_h5(data, filename, varname='data'):
    import h5py
    with h5py.File(filename, 'w') as f:
        f.create_dataset(varname, data=data)
        
        
def load_mat(filename, varname='data'):
    try:
        import scipy.io as sio
        f = sio.loadmat(filename)
        data = f[varname]        
    except:
        data = load_h5(filename, varname=varname)
        if data.ndim == 4:
            data = data.transpose(3,2,1,0)
        elif data.ndim == 3:
            data = data.transpose(2,1,0)
    return data
        
    
def load_dicom(foldername, flag_info=True):
    import pydicom
    import os
    foldername, _, filenames = next(os.walk(foldername))
    filenames = sorted(filenames)
    data, info = [], {}
    slice_min, loc_min, slice_max, loc_max = None, None, None, None
    for filename in filenames:
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername, filename))
        data.append(dataset.pixel_array)
        # Voxel size
        info['voxel_size'] = tuple(map(float, list(dataset.PixelSpacing) + [dataset.SpacingBetweenSlices]))
        # Slice location
        if slice_min is None or slice_min > float(dataset.SliceLocation):
            slice_min = float(dataset.SliceLocation)
            loc_min = np.array(dataset.ImagePositionPatient)
        if slice_max is None or slice_max < float(dataset.SliceLocation):
            slice_max = float(dataset.SliceLocation)
            loc_max = np.array(dataset.ImagePositionPatient)
    data = np.stack(data, axis=-1)
    # Matrix size
    info['matrix_size'] = data.shape
    # B0 direction
    affine2D = np.array(dataset.ImageOrientationPatient).reshape(2,3).T
    affine3D = (loc_max - loc_min) / ((info['matrix_size'][2]-1)*info['voxel_size'][2])
    affine3D = np.concatenate((affine2D, affine3D.reshape(3,1)), axis=1)
    info['B0_dir'] = tuple(np.dot(np.linalg.inv(affine3D), np.array([0, 0, 1])))
    if flag_info:
        return data, info
    else:
        return data
    
    
def save_dicom(data, foldername_tgt, foldername_src):
    import pydicom
    import os
    if not os.path.exists(foldername_tgt):
        os.mkdir(foldername_tgt)
    foldername_src, _, filenames = next(os.walk(foldername_src))
    filenames = sorted(filenames)
    for i, filename in enumerate(filenames):
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername_src, filename))
        dataset.PixelData = data[..., i].tobytes()
        dataset.save_as('{0}/{1}'.format(foldername_tgt, filename))



# Data processing
def resize_in_plane(inp):
    size = inp.shape
    out = np.zeros([size[0]//2, size[1]//2, size[2]])
    for i in range(0, size[0]//2):
        for j in range(0, size[1]//2):
            out[i, j, :] = (inp[2*i, 2*j, :] + inp[2*i+1, 2*j, :] + inp[2*i, 2*j+1, :] + inp[2*i+1, 2*j+1, :])/4
    return out


def extract_patches(img, patch_shape, extraction_step):
    from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
    patches = sk_extract_patches(img, patch_shape=patch_shape, extraction_step=extraction_step)
    ndim = img.ndim
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)


def rotate_xy(img, theta, voxel_size=(1,1,1)):
    '''
    In-plane (x-y) rotation by {theta} degree
    '''
    from scipy.interpolate import RegularGridInterpolator
    theta = theta / 180 * np.pi
    matrix_size = img.shape
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    FOV = np.array(matrix_size) * np.array(voxel_size)
    gridX = np.linspace(-0.5, 0.5, matrix_size[1])*FOV[1]
    gridY = np.linspace(-0.5, 0.5, matrix_size[0])*FOV[0]
    gridZ = np.linspace(-0.5, 0.5, matrix_size[2])*FOV[2]
    [X, Y, Z] = np.meshgrid(gridX, gridY, gridZ)
    loc = np.stack((Y.flatten(), X.flatten(), Z.flatten()), axis=1)
    loc_rot = np.dot(loc, rot.T);
    interp3 = RegularGridInterpolator((gridY, gridX, gridZ), img, method='nearest', bounds_error=False, fill_value=0)
    img_rot = interp3(loc_rot).reshape(matrix_size)
    return img_rot


def augment_data(img, voxel_size=(1,1,1), 
                      flip='xyz',
                      thetas=[]):
    '''
    Augment img by flipping and/or rotation
    Input:
        img,            ndarray(nx, ny, nz)
        voxel_size,     Tuple[int]
        flip,           str,                        axis for flipping
        thetas,         List[float],                angles for rotation (x-y plane)
    Output:
        imgs_aug,       ndarray(nsample, nx, ny, nz)
    '''
    imgs_aug = []
    imgs_aug.append(img) 
    if 'x' in flip:
        imgs_aug.append(np.flip(img, axis=0))
    if 'y' in flip:
        imgs_aug.append(np.flip(img, axis=1))
    if 'z' in flip:
        imgs_aug.append(np.flip(img, axis=2))
    for theta in thetas:
        imgs_aug.append(rotate_xy(img, theta, voxel_size=voxel_size))
    imgs_aug = np.stack(imgs_aug, axis=0)
    return imgs_aug
    


# Data reconstruction
def generate_indexes(img_size, patch_shape, extraction_step):
    import itertools
    ndims = len(patch_shape)
    # Patch center template
    #   [starts[i]                                                              starts[i]+patch_shape[i])
    #         [starts[i]+bound[i]        starts[i]+bound[i]+extraction_step[i])
    #   bound[i] = (patch_shape[i] - extraction_step[i])//2
    bound = [ (patch_shape[i] - extraction_step[i])//2 for i in range(ndims) ]
    npatches = [ (img_size[i] - patch_shape[i])//extraction_step[i] + 1 for i in range(ndims) ]
    
    starts = [ list(range(0, npatches[i]*extraction_step[i], extraction_step[i])) for i in range(ndims) ]
    ends = [ [ start+patch_shape[i] for start in starts[i] ] for i in range(ndims) ]
    starts_bound = [ [ start+bound[i] for start in starts[i] ] for i in range(ndims) ]
    ends_bound = [ [ start+bound[i]+extraction_step[i] for start in starts[i] ] for i in range(ndims) ]
    starts_local = [ [ bound[i] for start in starts[i] ] for i in range(ndims) ]
    ends_local = [ [ bound[i]+extraction_step[i] for start in starts[i] ] for i in range(ndims) ]
    
    # Extend to the edge of image
    for i in range(ndims):
        starts_bound[i][0] = 0
        starts_local[i][0] = 0
        ends_bound[i][-1] = ends[i][-1]
        ends_local[i][-1] = patch_shape[i]
    
    idxs = [ list(zip(starts_bound[i], ends_bound[i], starts_local[i], ends_local[i])) for i in range(ndims)]
    return itertools.product(*idxs)


def reconstruct_patches(patches, img_size, extraction_step, idxs_valid=None):
    npatches, *patch_shape = patches.shape
    if idxs_valid is None:
        idxs_valid = [True] * npatches
    count_valid = 0
    reconstructed_img = np.zeros(img_size, dtype=patches.dtype)
    for count, idx in enumerate(generate_indexes(img_size, patch_shape, extraction_step)):
        start_bound, end_bound, start_local, end_local = zip(*list(idx))
        selection_bound = [slice(start_bound[i], end_bound[i]) for i in range(len(idx))]
        selection_local = [slice(start_local[i], end_local[i]) for i in range(len(idx))]
        if idxs_valid[count]:
            reconstructed_img[selection_bound] = patches[count_valid][selection_local]
            count_valid += 1
    return reconstructed_img


        
# Plot
def plots(ims, figsize=(12,6), 
               rows=1, 
               scale=None, 
               interp=False, 
               titles=None):
    from matplotlib import pyplot as plt
    
    if scale != None:
        lo, hi = scale
        ims = ims.copy()
        ims[ims > hi] = hi
        ims[ims < lo] = lo
        ims = (ims - lo)/(hi - lo) * 1.0
        
    if ims.ndim == 2:
        ims = ims[np.newaxis, ..., np.newaxis];
    elif ims.ndim == 3:
        ims = ims[..., np.newaxis];
    ims = np.tile(ims, (1,1,1,3))
    #ims = ims.astype(np.uint8)
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

# dipole kernel in Fourier space
def dipole_kernel(matrix_size, voxel_size, B0_dir):
    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - ((X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2+0.0000000001)/(X**2 + Y**2 + Z**2+0.0000000001)# 防止D的分母到达0
    D[np.isnan(D)] = 0
    D = np.fft.fftshift(D);
    return D


## dipole kernle in image space

# def dipole_kernel(matrix_size, voxel_size, B0_dir): 
    
#     x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
#     y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
#     z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
#     Y, X, Z = np.meshgrid(x, y, z)
    
#     X = X*voxel_size[0]
#     Y = Y*voxel_size[1]
#     Z = Z*voxel_size[2]
    
#     d = (3*( X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2 - X**2-Y**2-Z**2)/(4*math.pi*(X**2+Y**2+Z**2)**2.5)

#     d[np.isnan(d)] = 0

#     return d


def sphere_kernel(matrix_size,voxel_size, radius):
    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X*voxel_size[0]
    Y = Y*voxel_size[1]
    Z = Z*voxel_size[2]
    
    Sphere_out = (np.maximum(abs(X) - 0.5*voxel_size[0], 0)**2 + np.maximum(abs(Y) - 0.5*voxel_size[1], 0)**2 
                  + np.maximum(abs(Z) - 0.5*voxel_size[2], 0)**2) > radius**2
    
    Sphere_in = ((abs(X) + 0.5*voxel_size[0])**2 + (abs(Y) + 0.5*voxel_size[1])**2 
                  + (abs(Z) + 0.5*voxel_size[2])**2) <= radius**2
    
    Sphere_mid = np.zeros(matrix_size)
    
    split = 10  #such that error is controlled at <1/(2*10)
    
    x_v = np.arange(-split+0.5, split+0.5, 1)
    y_v = np.arange(-split+0.5, split+0.5, 1)
    z_v = np.arange(-split+0.5, split+0.5, 1)
    X_v, Y_v, Z_v = np.meshgrid(x_v, y_v, z_v)
        
    X_v = X_v/(2*split)
    Y_v = Y_v/(2*split)
    Z_v = Z_v/(2*split)
    
    shell = 1-Sphere_in-Sphere_out
    X = X[shell==1]
    Y = Y[shell==1]
    Z = Z[shell==1]
    shell_val = np.zeros(X.shape)
    
    for i in range(X.size):
        xx = X[i]
        yy = Y[i]
        zz = Z[i]
        occupied = ((xx+X_v*voxel_size[0])**2+(yy+Y_v*voxel_size[1])**2+(zz+Z_v*voxel_size[2])**2)<=radius**2
        shell_val[i] = np.sum(occupied)/X_v.size
        
    Sphere_mid[shell==1] = shell_val
    Sphere = Sphere_in + Sphere_mid    
    Sphere = Sphere/np.sum(Sphere)
    y = np.fft.fftn(np.fft.fftshift(Sphere))
    return y
    
def SMV_kernel(matrix_size,voxel_size, radius):
    return 1-sphere_kernel(matrix_size, voxel_size,radius)

def SMV(iFreq,matrix_size,voxel_size,radius):
    return np.fft.ifftn(np.fft.fftn(iFreq)*sphere_kernel(matrix_size, voxel_size,radius))

def dataterm_mask(N_std, Mask, Normalize=True):
    w = Mask/N_std
    w[np.isnan(w)] = 0
    w[np.isinf(w)] = 0
    w = w*(Mask>0)
    if Normalize:
        w = w/np.mean(w[Mask>0])     
    return w

# directional differences
def dxp(a):
    return tf.concat((a[:,1:,:,:], a[:,-1:,:,:]), axis=1) - a

def dyp(a):
    return tf.concat((a[:,:,1:,:], a[:,:,-1:,:]), axis=2) - a

def dzp(a):
    return tf.concat((a[:,:,:,1:], a[:,:,:,-1:]), axis=3) - a

def fgrad(a, voxel_size):
    Dx = np.concatenate((a[1:,:,:], a[-1:,:,:]), axis=0) - a
    Dy = np.concatenate((a[:,1:,:], a[:,-1:,:]), axis=1) - a
    Dz = np.concatenate((a[:,:,1:], a[:,:,-1:]), axis=2) - a
    
    Dx = Dx/voxel_size[0]
    Dy = Dy/voxel_size[1]
    Dz = Dz/voxel_size[2]
    return np.concatenate((Dx[...,np.newaxis], Dy[...,np.newaxis], Dz[...,np.newaxis]), axis=3)

def gradient_mask(iMag, Mask, voxel_size=[1, 1, 3], percentage=0.9):
    field_noise_level = 0.01*np.max(iMag)
    wG = abs(fgrad(iMag*(Mask>0), voxel_size))
    denominator = np.sum(Mask[:]==1)
    numerator = np.sum(wG[:]>field_noise_level)
    
    if  (numerator/denominator)>percentage:
        while (numerator/denominator)>percentage:
            field_noise_level = field_noise_level*1.05
            numerator = np.sum(wG[:]>field_noise_level)
    else:
        while (numerator/denominator)<percentage:
            field_noise_level = field_noise_level*.95
            numerator = np.sum(wG[:]>field_noise_level)
            
    wG = (wG<=field_noise_level)
    return wG

def gradient_mask_patch(iMag, voxel_size=[1, 1, 3], percentage=0.9):
    field_noise_level = 0.01*tf.reduce_max(iMag)
    Mask = tf.cast(abs(iMag) > 0, tf.float32)
    Dx = dxp(iMag)/voxel_size[0]
    Dy = dyp(iMag)/voxel_size[1]
    Dz = dzp(iMag)/voxel_size[2]
    wG = abs(tf.concat([Dx[..., tf.newaxis], Dy[..., tf.newaxis], Dz[..., tf.newaxis]], axis=4))
    denominator = tf.reduce_sum(tf.cast(Mask[:]==1, tf.float32))
    numerator = tf.reduce_sum(tf.cast(wG[:]>field_noise_level, tf.float32))
    a = (numerator/denominator)>percentage
#     if  (numerator/denominator)>percentage:
#         while (numerator/denominator)>percentage:
#             field_noise_level = field_noise_level*1.05
#             numerator = tf.reduce_sum(tf.cast(wG[:]>field_noise_level, tf.float32))
#     else:
#         while (numerator/denominator)<percentage:
#             field_noise_level = field_noise_level*.95
#             numerator = tf.reduce_sum(tf.cast(wG[:]>field_noise_level, tf.float32))
            
    wG = tf.cast(wG<=field_noise_level, tf.float32)
    return wG
