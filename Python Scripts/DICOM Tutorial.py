import os
import pydicom
#from complexity_utilities import *
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans

dir = r'C:\Users\adamyarschenko\Documents\Research\Research_GitHub\Python Scripts'
os.chdir(dir)

data_path = r'\\TBMPFS\Research\AY\Exported Plans\Plan1'
output_path = working_path = r'\\TBMPFS\Research\AY\Exported Plans\Plan1\Plan 1 Output'
g = glob(data_path + '\*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))

#Find the dose file 'RD.*.dcm', load, and write contents to a text file. 
dose_path = [s for s in os.listdir(data_path) if s[0:3] == 'RD.']
dose_file = pydicom.filereader.dcmread(data_path + '\\' + dose_path[0])
with open('DICOM Contents.txt', 'w') as f:
    f.write(str(dose_file))
dose_cloud = dose_file.pixel_array

def load_scan(path):
    slices = [pydicom.filereader.dcmread(data_path + '\\' + s) for s in os.listdir(path) if s[0:3] == 'CT.']
    slices.sort(key = lambda x: int(x.InstanceNumber))
    if slices[0].SliceThickness == 0:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
        for s in slices:
            s.SliceThickness = slice_thickness
        
    return slices



def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=1
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

#Save the data to disk so you don't have to reprocess the stack every time.
np.save(output_path + '\\' + 'fullimages_%d.npy' % (id), imgs)

#Create histogram of the voxels in the study
file_used = output_path + '\\' + 'fullimages_{}.npy'.format(id)
imgs_to_process = np.load(file_used).astype(np.float64)
plt.hist(imgs_to_process.flatten(), bins = 500, color = 'c')
plt.xlim(-1500,1000)
plt.xlabel('Houndsfield Units (HU)')
plt.ylabel('Frequency')
plt.show()

#Plot an array of slices in a single plot, skipping ever 3 slices.
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows), int(i % rows)].set_title('slice{}'.format(ind))
        ax[int(i/rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i/rows), int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)

#Verify the slice thickness and voxel size. 
print('Slice thickness {}'.format(patient[0].SliceThickness))
print('Pixel spacing (row, col): (%f, %f)' % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
old_spacing = [patient[0].SliceThickness, patient[0].PixelSpacing]


#Reshape the voxels to a standardized size. 
def resample(image, scan, new_spacing):
    #Determine current pixel spacing.
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape*resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print('Shape before resampling\t', imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print('Shape after resampling\t', imgs_after_resamp.shape)

#Use the marching cubes algorithm to create a 3D plot of features in the images.
#Features are voxels matching a certain HU threshold such as bone or lung.
#Change the threshold to filter different voxels.
def make_mesh(image, threshold=-300, step_size=1): #Threshold was set to -300 for bone
    #Takes in resampled images and returns vertices and faces.
    print ("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces


#Make a static 3D plot with pyplot
def plt_3d(verts, faces):
    print ("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()

v, f = make_mesh(imgs_after_resamp, 350) #Was set to 350
plt_3d(v, f)

#Check images again after resampling. Choose single image for tests.
img = imgs_after_resamp[205]
plt.imshow(img)

#Start of function to create mask. 
#standardize the pixel values.
row_size = img.shape[0]
col_size = img.shape[1]

mean = np.mean(img)
std = np.std(img)
img = (img - mean)/std

#Mige want to adjust these values to get just the breast, or lung
# whichever mask you are trying to create.
middle = img[int(col_size/5):int(col_size/5*4), int(row_size/5):int(row_size/5*4)]
mean = np.mean(middle)
max = np.max(img)
min = np.min(img)

#Improving threshold finding by moving underflow and overflow
#on the pixel spectrum
img[img==max] = mean
img[img==min] = mean
plt.imshow(img)
#Separate soft tissue/bone from background lung/air using KMeans.
kmeans = KMeans(n_clusters = 2).fit(np.reshape(middle,[np.prod(middle.shape), 1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold, 1.0, 0.0)
plt.imshow(thresh_img)
#Erode finer elements, then dilate to include some pixels surrounding
#the segments you want. Don't want to accidently clip the segment.
eroded = morphology.erosion(thresh_img, np.ones([3,3]))
plt.imshow(eroded)
dilation = morphology.dilation(eroded, np.ones([8,8]))
plt.imshow(dilation)

labels = measure.label(dilation)
label_vals = np.unique(labels)
regions = measure.regionprops(labels)
good_labels = []
for prop in regions:
    B = prop.bbox
    if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
        good_labels.append(prop.label)
mask = np.ndarray([row_size,col_size],dtype=np.int8)
mask[:] = 0
#
#  After just the lungs are left, we do another large dilation
#  in order to fill in and out the lung mask 
#
for N in good_labels:
    mask = mask + np.where(labels==N,1,0)
mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation


fig, ax = plt.subplots(3, 2, figsize=[12, 12])
ax[0, 0].set_title("Original")
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')
ax[0, 1].set_title("Threshold")
ax[0, 1].imshow(thresh_img, cmap='gray')
ax[0, 1].axis('off')
ax[1, 0].set_title("After Erosion and Dilation")
ax[1, 0].imshow(dilation, cmap='gray')
ax[1, 0].axis('off')
ax[1, 1].set_title("Color Labels")
ax[1, 1].imshow(labels)
ax[1, 1].axis('off')
ax[2, 0].set_title("Final Mask")
ax[2, 0].imshow(mask, cmap='gray')
ax[2, 0].axis('off')
ax[2, 1].set_title("Apply Mask on Original")
ax[2, 1].imshow(mask*img, cmap='gray')
ax[2, 1].axis('off')
        
plt.show()