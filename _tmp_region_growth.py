#%% load libraries and images
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
unet_result = cv2.imread('pred.png', cv2.IMREAD_GRAYSCALE)
#%%
mask_ = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
mask_ = cv2.resize(mask_, img.shape)
#%%
# find edge of the unet result, which are the seed points for region growing
unet_edge_canny = cv2.Canny(unet_result, 100, 200)
# define unknown region
ret, thresh = cv2.threshold(unet_result, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
print('open:')
plt.imshow(opening)
#%%
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
print('close:')
plt.imshow(closing)
#%%
sure_bg = cv2.dilate(opening, kernel, iterations=1)
print('sure_bg:')
plt.imshow(sure_bg)
#%%
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
print('sure_fg:')
plt.imshow(sure_fg)
#%%
unknown = cv2.subtract(sure_bg, sure_fg)
print('unknown:')
plt.imshow(unknown)
#%%
unet_edge = np.bitwise_and(unet_result > 10, unet_result < 250)
print('unet_edge:')
plt.imshow(unet_edge)
#%% define region growing function
def region_growing(img, seeds, threshold=2):
    seeds = np.array(np.argwhere(seeds > 0), dtype=np.int32)
    output = np.zeros_like(img)
    img_copy = img.copy()
    diff = threshold
    for seed in seeds:
        seed[0], seed[1] = seed[1], seed[0]# np to cv2
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img_copy,
                      mask,
                      tuple(seed),
                      255,
                      loDiff=diff,
                      upDiff=diff)
        output[mask[1:-1,1:-1] > 0] += 5
    return output

#%%
# do region growing
result = region_growing(img, unet_edge_canny)
print('region growth result:')
plt.imshow(result)
#%% add to the unet result
fin_result = cv2.add(result, unet_result)
print('final result:')
plt.imshow(fin_result)
#%%
# read truth
truth = cv2.imread('true.png', cv2.IMREAD_GRAYSCALE)
print('truth:')
plt.imshow(truth)
# %%
# top-hat transform
kernel = np.ones((3, 3), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
white_tophat = cv2.add(tophat, img)
black_tophat = cv2.subtract(img, tophat)
# print('tophat:')
plt.imshow(tophat)
print('white tophat:')
# plt.imshow(white_tophat)
# print('black tophat:')
# plt.imshow(black_tophat)
# %% x
# do region growing
result = region_growing(white_tophat, unet_edge_canny)
print('region growth result:')
plt.imshow(result)
# add to the unet result
fin_result = cv2.add(result, unet_result)
print('final result:')
plt.imshow(fin_result)
#%% erode unet result
kernel = np.ones((2, 2), np.uint8)
erosion = cv2.erode(unet_result, kernel, iterations=1)
print('erosion:')
plt.imshow(erosion)
#%%
# do region growing
result = region_growing(black_tophat, erosion, 2)
print('region growth result:')
plt.imshow(result)
#%%
erosion_img = cv2.add(img, erosion)
print('erosion_img:')
plt.imshow(erosion_img)
#%%
def queue_region_growth(img, seeds, threshold=2):
    seeds = np.array(np.argwhere(seeds > 0), dtype=np.int32)
    output = np.zeros_like(img)
    diff = threshold
    around = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    explored = np.zeros_like(img)
    for seed in seeds:
        seed[0], seed[1] = seed[1], seed[0]# np to cv2
        explored[seed[0], seed[1]] = 1
        # init queue
        queue = [seed + around[i] for i in range(around.shape[0])]
        queue_seed = [seed]*8
        while len(queue) > 0:
            current_point = queue.pop(0)
            current_seed = queue_seed.pop(0)
            if current_point[0] < 0 or current_point[1] < 0 or current_point[0] >= img.shape[0] or current_point[1] >= img.shape[1] or explored[current_point[0], current_point[1]] >= 8:
                continue
            explored[current_point[0], current_point[1]] += 1
            if np.abs(int(img[current_point[0], current_point[1]]) - int(img[current_seed[0], current_seed[1]])) < diff:
                output[current_point[0], current_point[1]] = 255
                queue.extend([current_point + around[i] for i in range(around.shape[0])])
                queue_seed.extend([current_seed] * 8)
    return output
#%%
# do region growing
result = queue_region_growth(img, erosion, 2)
print('region growth result:')
# plt.imshow(result)
plt.imshow(cv2.add(result, unet_result))
#%%
final_result = cv2.add(result, unet_result)
# gaussian blur
blur = cv2.GaussianBlur(final_result, (3, 3), 0)
print('blur:')
plt.imshow(blur)
#%%
masked_blur_final_result = cv2.bitwise_and(blur, mask_)
masked_blur_final_result[masked_blur_final_result > 150] = 255
print('masked_blur_final_result:')
plt.imshow(masked_blur_final_result)
cv2.imwrite('masked_blur_final_result.jpg', masked_blur_final_result)