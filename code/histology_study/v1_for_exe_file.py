# When creating an exe file using auto-py-to-exe, I need to include the following folder:
# C:\ProgramData\Miniconda3\envs\test_gui\Lib\site-packages\pyclesperanto_prototype

# %% #Import necessary libraries
from numpy import zeros_like, stack, percentile, dstack, invert
from matplotlib.pyplot import subplots, savefig, rcParams

from skimage import data, io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity

# from pyclesperanto_prototype import voronoi_otsu_labeling, statistics_of_labelled_pixels, imshow   #main algorithm is HERE!!!
import pyclesperanto_prototype as cle

from pandas import DataFrame
# %% Convert RGB into HED (to find only brown cells)
def color_separate(ihc_rgb):

    ihc_hed = rgb2hed(ihc_rgb)
    null = zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(stack((ihc_hed[:,:,0], null, null), axis = -1)))
    ihc_e = img_as_ubyte(hed2rgb(stack((null, ihc_hed[:,:,1], null), axis = -1)))
    ihc_d = img_as_ubyte(hed2rgb(stack((null, null, ihc_hed[:,:,2]), axis = -1)))

    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                            in_range = (0, percentile(ihc_hed[:,:,0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                            in_range = (0, percentile(ihc_hed[:,:,2], 99)))

    zdh = img_as_ubyte(dstack((null, d, h)))
    return (ihc_h, ihc_e, ihc_d, zdh)

# %%
CRED = ''
CEND = ''
try:
    new_img = input('Provide a name for an image(*.jpg/png) IN THE SAME DIRECTORY AS exe file: ')
    # new_img = "test.jpg"
    #new_img = resource_path(new_img)
    ihc_rgb = io.imread(new_img)
    # %%
    H, E, D, HD = color_separate(ihc_rgb)
    # %% Make dark background and bright cells
    input_image = invert(D[:, :, 2])
    input_image_H = invert(H[:, :, 2])
    # plt.imshow(input_image, cmap='gray')

    # %% Main algorithm for image segmentation

    # device = cle.select_device(cle.available_device_names(dev_type='gpu')[0])
    # device
    # input_gpu = cle.push(input_image)
    input_gpu = input_image
    input_gpu_H = input_image_H

    #IMPORTANT PARAMETERS THAT CAN BE TUNED BY USER
    # sigma_spot_detection = 3
    # sigma_outline = 1

    sigma_spot_detection = abs(float(input('Provide parameter 1 [how close cells can be], you can start with VAL=3, and tune it later: ')))
    sigma_outline = abs(float(input('Provide parameter 2 [how precise objects are segmented], you can start with VAL=1, and tune it later: ')))
    print('\n ---------------------PLEASE WAIT, RESULTS ARE COMING!-----------------------')
    segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=sigma_spot_detection, 
                                        outline_sigma=sigma_outline)
    segmented_H = cle.voronoi_otsu_labeling(input_gpu_H, spot_sigma=sigma_spot_detection, 
                                        outline_sigma=sigma_outline)

    # %%
    statistics = cle.statistics_of_labelled_pixels(input_gpu, segmented) 
    table = DataFrame(statistics)   
    # table_2 = table.describe()
    # table_2.to_excel('statistics.xlsx')
    # print('\n----------Excel file with stat information is created-------------')
    # %%
    statistics_H = cle.statistics_of_labelled_pixels(input_gpu_H, segmented_H) 
    table_H = DataFrame(statistics_H)   
    # table_2 = table.describe()
    # table_2.to_excel('statistics.xlsx')
    # print('\n----------Excel file with stat information is created-------------')
    # %% Final results
    rcParams['figure.figsize'] = [20,20]
    fig, axs = subplots(1,3)
    fig.set_dpi(200)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[0].title.set_text('Original')
    axs[1].title.set_text('Brown')
    axs[2].title.set_text('Segmented')
    axs[0].imshow(ihc_rgb)
    axs[1].imshow(D)
    cle.imshow(segmented, labels=True, plot=axs[2])

    savefig('Decomposed.jpg')
    final_df = DataFrame(data = {'Brown':[int(table.describe().label[0])], 'Blue':[int(table_H.describe().label[0])]})
    final_df.to_csv('N_cells_Par1_'+str(sigma_spot_detection)+'_Par2_'+str(sigma_outline)+'.csv', sep='\t', index=None)

    print(CRED+'\n------------Decomposed image is saved as Decomposed.jpg---------------'+CEND)

    print(CRED+'\n------------Info about number of cells is saved as N_cells_YOUR_PARAMETERS_USED.csv---------------'+CEND)

    print(CRED+'\n----------------------Number of brown cells is ', int(table.describe().label[0]), '-----------------'+CEND)

    print(CRED+'\n----------------------Number of blue cells is ', int(table_H.describe().label[0]), '-----------------'+CEND)

    input("Press enter to exit")
except:
    print(CRED+"ERROR: either no such file, or your input parameter is not applicable"+CEND)