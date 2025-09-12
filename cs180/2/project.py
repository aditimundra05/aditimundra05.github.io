# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.transform as sktransform
import skimage.filters as skfilters

def process(sliding_window=15):
    # name of the input file
    imname = 'cathedral.jpg'

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    one, two, ag = pyramid_speedup(g, b, sliding_window)
    three, four, ar = pyramid_speedup(r, b, sliding_window)
    # create a color image
    im_out = np.dstack([ar, ag, b])
    im_out_uint8 = sk.img_as_ubyte(im_out)

    # save the image
    fname = 'new_cathedral.jpg'
    print("Green Blue Alignment: " + str(one) + ", " + str(two))
    print("Red Blue Alignment: " + str(three) + ", " + str(four))
    skio.imsave(fname, im_out_uint8)

    # display the image
    skio.imshow(im_out_uint8)
    skio.show()

def align(first, second, sliding_window):
    score = -float('inf')
    x, y = 0, 0

    for dx in range(-sliding_window, sliding_window + 1):
        for dy in range(-sliding_window, sliding_window + 1):
            aligned = np.roll(first, (dx, dy), axis=(0, 1))
            this_score = ncc(second, aligned)
            if this_score > score:
                x, y = dx, dy
                score = this_score

    aligned_img = np.roll(first, (x, y), axis=(0,1))
    return x, y, aligned_img

def pyramid_speedup(first, second, sliding_window, levels=5):
    ref = [second]
    transform = [first]
    x, y = 0, 0

    for level in range(1, levels):
        # downscale the image
        ref_down = sktransform.rescale(ref[-1], 0.5, anti_aliasing=True, channel_axis=None)
        transform_down = sktransform.rescale(transform[-1], 0.5, anti_aliasing=True, channel_axis=None)
        
        ref.append(ref_down)
        transform.append(transform_down)
    
    for level in range(levels - 1, -1, -1):
        if level < levels - 1:
            x *= 2
            y *= 2
        
        curr_sliding_window = max(1, sliding_window // (2 ** level))
        curr_img = np.roll(transform[level], (x, y), axis=(0, 1))
        dx, dy, _ = align(curr_img, ref[level], curr_sliding_window)
        x += dx
        y += dy

    aligned_img = np.roll(first, (x, y), axis=(0,1))
    return x, y, aligned_img
    
def l2norm(u, v):
    # euclidean distance
    return np.sqrt(np.sum((u - v) ** 2))

def ncc(u, v):
    u_norm = u - np.mean(u)
    v_norm = v - np.mean(v)
    bottom = np.sqrt(np.sum(u_norm ** 2) * np.sum(v_norm ** 2))
    if bottom == 0:
        return 0
    return np.sum(u_norm * v_norm) / bottom


if __name__ == "__main__":
    process(15)