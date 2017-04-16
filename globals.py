HOG_HSV = { 'color_space':'HSV', 
        'hog_orient' : 9,  
        'hog_pix_per_cell' : 8, 
        'hog_cell_per_block' : 4, 
        'hog_channel' : 2 # or 0, 1, 2
        }

HOG_LUV = { 'color_space':'LUV', 
        'hog_orient' : 8,  
        'hog_pix_per_cell' : 8, 
        'hog_cell_per_block' : 2, 
        'hog_channel' : 0 # or 0, 1, 2
        }

HOG_RGB = { 'color_space':'RGB', 
        'hog_orient' : 30,  
        'hog_pix_per_cell' : 16, 
        'hog_cell_per_block' : 2, 
        'hog_channel' : 'ALL' # or 0, 1, 2
        }

hog_spatial_size = (16, 16)
hog_hist_bins = 16
spatial_feat = True 
hist_feat = True 
hog_feat = True 
minArea = 6400  
overlap = 0.75
memory = 30

#windowSizes=[128,96,128]
#xy_start = [(150,400),(500,400), (950,400)]
#xy_stop = [(400,600),(1150,460), (1240,600)]

windowSizes=[96,128]
xy_start = [(500,400), (950,400)]
xy_stop = [(1150,460), (1240,600)]
