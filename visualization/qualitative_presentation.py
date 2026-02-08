# %% [markdown]
# PhD team 
# Presenting Code for PhD Thesis

# %% info [markdown]
# Here, we are presenting a completed routine, a thorough connected blocks, to perform the very idea of my Ph.D. thesis, conducting a comprehensively automatic and longitudinal analyses of a given MS patient.
# All of the rights of this routine are reserved for the developer(s).
# 
# 
# Mahdi Bashiri Bawil
# Developer

# %% [markdown]
# 

# %% [markdown]
# # Attempt : Comprehensive Analyses 

# %% [markdown]
# %% [markdown]
# ## Phase 0: Dependencies & Functions

# %% [markdown]
# ### Packages

# %% Packages
import os
import cv2
import skimage
import numpy as np
from PIL import Image
from skimage.color import gray2rgb
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


# %% Qualitative Presentation

#
def outliner(m_data):

    m_data = (np.where(m_data > 0, 1, 0)).astype(bool)
    dilated_data = binary_dilation(m_data, structure=disk(1)[np.newaxis, ...])
    outlined_data = dilated_data & ~m_data
    o_data = (outlined_data * 1).astype(np.float32)

    return o_data

#
def rgb_encoder(rgb_code):

    out_code = np.array([0, 0, 0])
    
    out_code[0] = int(rgb_code.split(',')[0][4:])
    out_code[1] = int(rgb_code.split(',')[1][:])
    out_code[2] = int(rgb_code.split(',')[2][:-1])

    return out_code

#
def slice20(stacks):

    stacks_tmp = np.zeros((np.shape(stacks)[0], np.shape(stacks)[1], 20))    # to ensure having always 20 slices
    
    stack_max = np.max(stacks)
    stack_max = 1 if stack_max == 0 else stack_max
    for c in range(0, np.shape(stacks)[-1]):
        if c == 20:             # to avoid more than 20 slices of some subjects!
            break

        stacks_tmp[..., c] = stacks[..., c] / stack_max


    return stacks_tmp

#
def big_picture_old(stacks):

    stacks = stacks[13:-13, 13:-13, ...]      
    shapee = np.shape(stacks)

    temp = np.zeros((2 * shapee[0], 2 * shapee[1], shapee[2], shapee[3]))
    for c in range(0, shapee[3]):
        temp[..., c] = cv2.resize(stacks[..., c], (460, 460))
    stacks = temp

    rows = 4
    columns = 5
    image_n = 2 * 230
    image_m = 2 * 230
    image = np.zeros((image_n * rows, image_m * columns, 3))

    # get the background color:
    back = stacks[10:20, 20:100, 0, -1]
    stacks = np.where(stacks < 0.1 * np.max(stacks), np.min(stacks), stacks)        #
    back = np.average(back)
    image += back

    for i in range(0, np.shape(stacks)[3]):
        fl = np.uint8(stacks[..., i] * 255)
        ff = fl

        if i < columns:
            image[0:image_n, image_m * i: image_m * (i+1), :] = ff
        elif columns <= i < 2 * columns:
            image[image_n:image_n * 2, image_m * (i-columns): image_m * (i-columns+1), :] = ff
        elif 2 * columns <= i < 3 * columns:
            image[image_n * 2:image_n * 3, image_m * (i-2*columns): image_m * (i-2*columns+1), :] = ff
        elif 3 * columns <= i < 4 * columns:
            image[image_n * 3:image_n * 4, image_m * (i-3*columns): image_m * (i-3*columns+1), :] = ff


    return np.round((image / np.max(image) * 255.0)).astype(np.uint8)

#
def big_picture(stacks):

    paper_height = 9; paper_width = 16         # as of 16:9 screen size
    paper_ratio = paper_width / paper_height
    slices = stacks.shape[-1]

    # calculate needed columns and rows
    x = np.sqrt(slices / paper_ratio)
    y = paper_ratio * x

    columns = np.ceil(y).astype(np.uint16)
    rows = np.ceil(x).astype(np.uint16)

    image_size = min(np.floor(((paper_width) * 300) / columns), np.floor(((paper_height) * 300) / rows)).round().astype(np.uint16)


    # crop the images to focus on the brain
    stacks = stacks[13:-13, 13:-13, ...]      
    shapee = np.shape(stacks)

    temp = np.zeros((image_size, image_size, shapee[2], shapee[3]))
    for c in range(0, shapee[3]):
        temp[..., c] = cv2.resize(stacks[..., c], (image_size, image_size))
    stacks = temp

    canvas = np.zeros((image_size * rows, image_size * columns, 3))

    # get the background color:
    stacks = np.where(stacks < 0.1 * np.max(stacks), np.min(stacks), stacks)        #
    back = stacks[10:20, 20:100, 0, -1]
    back = np.average(back)
    canvas += back


    slc_den = 0
    for n in range(rows):
        for m in range(columns):

            if slc_den == stacks.shape[-1]:
                break

            to_canvas = stacks[..., slc_den]
            canvas[image_size * n: image_size * (n+1), image_size * m: image_size * (m+1), :] = to_canvas

            slc_den = slc_den + 1  

    return ((canvas) * 255).astype(np.uint8)

#
def big_picture2(stacks, raw_stacks):

    paper_height = 11.7; paper_width = 8.27         # as of A4 paper
    paper_ratio = paper_width / paper_height
    slices = stacks.shape[-1]

    # calculate needed columns and rows
    x = np.sqrt(2 * slices / paper_ratio)
    y = paper_ratio * x

    columns = np.ceil(y).astype(np.uint16)
    rows = np.ceil(x).astype(np.uint16)

    image_size = min(np.floor(((paper_width - 0.5) * 300) / columns), np.floor(((paper_height - 0.5) * 300) / rows)).round().astype(np.uint16)


    # crop the images to focus on the brain
    stacks = stacks[13:-13, 13:-13, ...]      
    raw_stacks = raw_stacks[13:-13, 13:-13, ...]      
    shapee = np.shape(stacks)

    temp = np.zeros((image_size, image_size, shapee[2], shapee[3]))
    temp_raw = np.copy(temp)
    for c in range(0, shapee[3]):
        temp[..., c] = cv2.resize(stacks[..., c], (image_size, image_size))
        temp_raw[..., c] = cv2.resize(raw_stacks[..., c], (image_size, image_size))
    stacks = temp
    raw_stacks = temp_raw

    canvas = np.zeros((image_size * rows, image_size * columns, 3))

    # get the background color:
    stacks = np.where(stacks < 0.1 * np.max(stacks), np.min(stacks), stacks)        #
    raw_stacks = np.where(raw_stacks < 0.1 * np.max(raw_stacks), np.min(raw_stacks), raw_stacks)        #
    back = stacks[10:20, 20:100, 0, -1]
    back = np.average(back)
    canvas += back


    slc_den = 0
    for n in range(rows):
        for m in range(columns):

            if slc_den == (2 * stacks.shape[-1]):
                break

            # Select the appropriate image slice and place the slice on the canvas
            to_canvas = raw_stacks[..., (slc_den // 2)] if (slc_den % 2)==0 else stacks[..., ((slc_den - 1) // 2)]
            canvas[image_size * n: image_size * (n+1), image_size * m: image_size * (m+1), :] = to_canvas 
                                                                                                 
            slc_den = slc_den + 1  

    return ((canvas) * 255).astype(np.uint8)

#
def specific_result(flair_data, vent_masks, csf_masks, jv_mask, save_path, sub_id, tp, wmh_analysis_path):

    # main flair:
    flair_main = np.copy(flair_data)

    # WMH:
    files = os.listdir(wmh_analysis_path)
    # print(files)
    for file in files:
        if file[-7:] != '.pickle':
            continue
        with open(os.path.join(wmh_analysis_path, file), 'rb') as handle:
            res = pickle.load(handle)
        # if file[:-7] == 'flair_new':
        #     flair_f = res
        if file[:-7] == 'area':
            area = res
        elif file[:-7] == 'code':
            code = res
        elif file[:-7] == 'peri_mask':
            peri = res
        elif file[:-7] == 'para_mask':
            para = res
        elif file[:-7] == 'juxt_mask':
            juxt = res

    # Loading data:
    peri_data = peri.astype(np.uint8)
    para_data = para.astype(np.uint8)
    juxt_data = juxt.astype(np.uint8)
    jvent_data = np.where(jv_mask > 0, 1, 0).astype(np.uint8)
    vent_data = np.where(vent_masks > 0, 1, 0).astype(np.uint8)
    csf_data = np.where(csf_masks > 0, 1, 0).astype(np.uint8)

    # Save the specific slices:     
    """    
    n_slc_ = 13    
    output_path_sp = os.path.join(save_path, f'csf_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(csf_data[35:-35, 35:-35, n_slc_-1] * 255))

    output_path_sp = os.path.join(save_path, f'vent_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(vent_data[35:-35, 35:-35, n_slc_-1] * 255))
    
    output_path_sp = os.path.join(save_path, f'jvent_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(jvent_data[35:-35, 35:-35, n_slc_-1] * 255))

    output_path_sp = os.path.join(save_path, f'ab_wmh_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8((peri_data[35:-35, 35:-35, n_slc_-1] + para_data[35:-35, 35:-35, n_slc_-1] + juxt_data[35:-35, 35:-35, n_slc_-1]) * 255))

    output_path_sp = os.path.join(save_path, f'peri_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(peri_data[35:-35, 35:-35, n_slc_-1] * 255))

    output_path_sp = os.path.join(save_path, f'para_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(para_data[35:-35, 35:-35, n_slc_-1] * 255))
    
    output_path_sp = os.path.join(save_path, f'juxt_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(juxt_data[35:-35, 35:-35, n_slc_-1] * 255))
    """

    # peri, para, juxt outlines:
    # peri_outline = outliner(peri_data)
    # para_outline = outliner(para_data)
    # juxt_outline = outliner(juxt_data)
    # jvent_outline = outliner(jvent_data)

    # Fixating the FLAIR Images onto a 20-slices stack:
    # flair_main = slice20(flair_main)
    # peri = slice20(peri)
    # para = slice20(para)
    # juxt = slice20(juxt)
    # jvent_data = slice20(jvent_data)
    # vent = slice20(vent)

    # Convert the main flair to RGB
    flair_rgb = gray2rgb(flair_main)  # Converts grayscale flair_main to RGB format
    flair_rgb = np.transpose(flair_rgb, (0, 1, 3, 2)) / np.max(flair_rgb)       # Map to [0, 1]


    # Create RGB representations for the outlines
    peri_outline_rgb = np.zeros_like(flair_rgb)
    para_outline_rgb = np.zeros_like(flair_rgb)
    juxt_outline_rgb = np.zeros_like(flair_rgb)
    jvent_outline_rgb = np.zeros_like(flair_rgb)
    vent_outline_rgb = np.zeros_like(flair_rgb)
    csf_outline_rgb = np.zeros_like(flair_rgb)

    peri_color = rgb_encoder(color_codes['peri'])
    peri_outline_rgb[..., 0, :] = (peri_color[0] / 255) * peri_data
    peri_outline_rgb[..., 1, :] = (peri_color[1] / 255) * peri_data
    peri_outline_rgb[..., 2, :] = (peri_color[2] / 255) * peri_data

    para_color = rgb_encoder(color_codes['para'])
    para_outline_rgb[..., 0, :] = (para_color[0]/255) * para_data
    para_outline_rgb[..., 1, :] = (para_color[1]/255) * para_data
    para_outline_rgb[..., 2, :] = (para_color[2]/255) * para_data

    juxt_color = rgb_encoder(color_codes['juxt'])
    juxt_outline_rgb[..., 0, :] = (juxt_color[0]/255) * juxt_data
    juxt_outline_rgb[..., 1, :] = (juxt_color[1]/255) * juxt_data
    juxt_outline_rgb[..., 2, :] = (juxt_color[2]/255) * juxt_data

    jvent_color = rgb_encoder(color_codes['jvent'])
    jvent_outline_rgb[..., 0, :] = (jvent_color[0]/255) * jvent_data
    jvent_outline_rgb[..., 1, :] = (jvent_color[1]/255) * jvent_data
    jvent_outline_rgb[..., 2, :] = (jvent_color[2]/255) * jvent_data

    vent_color = rgb_encoder(color_codes['vent'])
    vent_outline_rgb[..., 0, :] = (vent_color[0]/255) * vent_data
    vent_outline_rgb[..., 1, :] = (vent_color[1]/255) * vent_data
    vent_outline_rgb[..., 2, :] = (vent_color[2]/255) * vent_data

    # Combine all color masks to make a united one
    color_masks = peri_outline_rgb + para_outline_rgb + juxt_outline_rgb  + vent_outline_rgb + jvent_outline_rgb

    # Post-process the csf_mask to exclude the vent_mask from that, to avoid overlapping
    csf_mask_bool = np.where(csf_data > 0, 1, 0).astype(bool)
    all_masks = np.mean(color_masks, axis=2)
    wmh_vent_bool = np.where(all_masks > 0, 1, 0).astype(bool)
    csf_bool = csf_mask_bool & ~wmh_vent_bool
    csf_mask_exc = np.copy(csf_data)
    csf_mask_exc[csf_bool == False] = 0 
    
    csf_color = rgb_encoder(color_codes['csf'])
    csf_outline_rgb[..., 0, :] = (csf_color[0]/255) * csf_mask_exc
    csf_outline_rgb[..., 1, :] = (csf_color[1]/255) * csf_mask_exc
    csf_outline_rgb[..., 2, :] = (csf_color[2]/255) * csf_mask_exc

    color_masks += csf_outline_rgb

    # Filter out FLAIR image

    color_masks_bool = np.where(color_masks > 0, 1, 0).astype(bool)
    # Perform the OR operation across the third axis (axis 2)
    color_masks_or = np.max(color_masks_bool, axis=2)
    expanded_color_masks_or = np.expand_dims(color_masks_or, axis=2)  # Shape becomes (256, 256, 1, 20)
    # Repeat the values across the third axis (color channels)
    expanded_color_masks_or = np.repeat(expanded_color_masks_or, 3, axis=2)  # Shape becomes (256, 256, 3, 20)

    flair_bool = np.where(flair_rgb > 0, 1, 0).astype(bool)
    flair_bool = flair_bool & ~expanded_color_masks_or
    flair_rgb_filtered = np.copy(flair_rgb)
    flair_rgb_filtered[flair_bool == False] = 0

    # # Blind the FLAIR image with color masks
    flair_rgb_masked = flair_rgb_filtered + color_masks
    flair_rgb_masked_whole = big_picture2(flair_rgb_masked, flair_rgb)

    output_path = os.path.join(save_path, 'All_Slices_' + str(sub_id) + '_tp' + str(tp) + '.png')
    skimage.io.imsave(output_path, flair_rgb_masked_whole)

    # Special Save:
    """
    output_path_sp = os.path.join(save_path, f'img_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(flair_rgb_masked[35:-35, 35:-35,:, n_slc_-1] * 255))
    output_path_sp = os.path.join(save_path, f'raw_slice{n_slc_}.png')
    skimage.io.imsave(output_path_sp, np.uint8(flair_rgb[35:-35, 35:-35,:, n_slc_-1] * 255))
    """

    ## Make a GIF image from the two images

    # make a picture of processed flair images and color masks
    flair_whole = big_picture(flair_rgb)
    flair_rgb_masked_whole = big_picture(flair_rgb_masked)

    # Convert numpy arrays to PIL images
    pil_image1 = Image.fromarray(flair_whole, 'RGB')
    pil_image2 = Image.fromarray(flair_rgb_masked_whole, 'RGB')

    # Save as a GIF
    output_path = os.path.join(save_path, '1.png')
    pil_image1.save(output_path)
    output_path = os.path.join(save_path, '2.png')
    pil_image2.save(output_path)

    output_path = os.path.join(save_path, 'gAll_Slices_' + str(sub_id) + '_tp' + str(tp) + '.gif')
    pil_image1.save(output_path, save_all=True, append_images=[pil_image2], duration=1000, loop=0)

    return

#
def diff(value2, value1):
    value1 = np.float64(value1)
    value2 = np.float64(value2)
    dif = np.round((value2 - value1) / value1 * 100, 1)
    if dif < 0:
        sign = '-'
        clr = (255, 0, 51)
    else:
        sign = '+'
        clr = (0, 204, 102)
    dif = np.abs(dif)
    return dif, sign, clr

