import numpy as np 
import math
import matplotlib.pyplot as plt 
from PIL import Image 
from matplotlib.colors import colorConverter
import matplotlib as mpl

# maximum angle of slope (degrees)
max_angle =12.5
radian_value = np.radians(max_angle)
G = np.tan(radian_value)


# Sample.jpg assigned to data_source variable 
data_source = Image.open('sample.jpg')

# Converts data_source to greyscale (0-255) and converts data_source to a numpy array 
data_source = data_source.convert('L')
data_source = np.array(data_source)

# Retrieves gradients in x and y dimensions and stores results in numpy arrays 
array_1, array_2= np.gradient(data_source, 59)
array_1 = np.array(array_1)
array_2 = np.array(array_2)

# This is where the range of acceptable gradients is defined (for each dimension)
resultarray_1 = np.where(np.logical_and(array_1>=-G, array_1<=G))
resultarray_2 = np.where(np.logical_and(array_2>=-G, array_2<=G))

# Numpy zeros array must be the same shape as the input data_source 
one_plot_array_1 = np.zeros(np.array(data_source).shape)
one_plot_array_2 = np.zeros(np.array(data_source).shape)

# Indexing the np.zeros arrays to equal 1 where the conditions are met
one_plot_array_1[resultarray_1] = 1
one_plot_array_2[resultarray_2] = 1

# Stating the conditions needed to compare one_plot_array_1 and one_plot_array_2 
c1 = (one_plot_array_1 == 1)
c2 = (one_plot_array_2 == 1)

# Retrieving the location where both conditions c1 and c2 are met 
result1 = np.where(c1 & c2)

# Creating an array that displays a 1 where the gradients are within the specified range
one_plot_ = np.zeros(np.array(data_source).shape)
one_plot_[result1] = 1

# Converting data_source type to uint8
finalplot = one_plot_.astype(np.uint8)
finalplot = np.flip(finalplot,0)
crop = finalplot[170:245,360:435]
np.save('environment.npy', crop)
data_source = np.flip(data_source,0)


# create dummy data_source
zvals = data_source
zvals2 = finalplot

# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('green')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init()
# alphas = np.linspace(0, 0.8, cmap2.N+3)
alphas = np.linspace(0, 0.5, cmap2.N+3)
cmap2._lut[:,-1] = alphas


img2 = plt.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
img3 = plt.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')

plt.show()

