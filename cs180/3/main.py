import numpy as np
from skimage import io, color, exposure
import skimage
from scipy import signal
import cv2
import math
import matplotlib.pyplot as plt
import skimage.transform as sktr

def convolution_four_loops(img, filter):
    h, w = img.shape
    fh, fw = filter.shape
    filter = np.flipud(np.fliplr(filter))

    output = np.zeros((h, w))
    padded = np.zeros((h + 2 * (fh // 2), w + 2 * (fw // 2)))
    padded[(fh // 2) : (fh // 2) + h, (fw // 2) : (fw // 2) + w] = img
    for x in range(h):
        for y in range(w):
            conv = 0.0
            for fx in range(fh):
                for fy in range(fw):
                    conv += filter[fx, fy] * padded[x + fx, y + fy]
            output[x, y] = conv
    
    return np.clip(output, 0, 1)
                    
def convolution_two_loops(img, filter):
    h, w = img.shape
    fh, fw = filter.shape
    filter = np.flipud(np.fliplr(filter))

    output = np.zeros((h, w))
    padded = np.zeros((h + 2 * (fh // 2), w + 2 * (fw // 2)))
    padded[(fh // 2) : (fh // 2) + h, (fw // 2) : (fw // 2) + w] = img
    for x in range(h):
        for y in range(w):
            conv = padded[x:x+fh, y:y+fw]
            output[x, y] = np.sum(conv * filter)
    
    return output

def gradients(image, dx, dy):
    grad_x = signal.convolve2d(image, dx, mode='same', boundary='symm')
    grad_y = signal.convolve2d(image, dy, mode='same', boundary='symm')
    return grad_x, grad_y

def gradient_mag(grad_x, grad_y):
    return np.sqrt(grad_x**2 + grad_y**2)

def threshold(grad_mag, threshold):
    return (grad_mag > threshold).astype(float)

def create_gaussian(sigma):
    n = int(2*np.ceil(3*sigma) + 1)
    gaussian_oned = cv2.getGaussianKernel(n, sigma)
    return gaussian_oned @ gaussian_oned.T

def unsharp_mask_filter():
    G = create_gaussian(1)
    alpha = 1.5
    center = G.shape[0] // 2
    e = np.zeros_like(G)
    e[center, center] = 1.0
    return ((1 + alpha) * e - alpha * G)

def apply_unsharp_mask(image, sigma=1):
    unsharp_kernel = unsharp_mask_filter()
    sharpened = np.zeros_like(image)
    blurred = gaussian_filter(image, sigma)
    for c in range(image.shape[2]):
        sharpened[:, :, c] = signal.convolve2d(image[:, :, c], unsharp_kernel, mode='same', boundary='symm')
    high_freq = image - blurred
    return np.clip(sharpened, 0, 1), blurred, high_freq

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=-1)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=-1)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def gaussian_filter(image, sigma):
    kernel = create_gaussian(sigma) 
    if len(image.shape) == 2:
        return signal.convolve2d(image, kernel, mode='same', boundary='symm')   
    filtered = np.zeros_like(image)
    for c in range(image.shape[2]):
        filtered[:, :, c] = signal.convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
    return filtered

def hybrid_image(im1, im2, sigma1, sigma2):
    low_pass = gaussian_filter(im2, sigma2)
    high_pass = im1 - gaussian_filter(im1, sigma1)
    return low_pass, high_pass, np.clip(low_pass + high_pass, 0, 1)

def show_frequency_analysis(original1, original2, low_pass, high_pass, hybrid):    
    def compute_fft_log(image):
        if len(image.shape) == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))))

    fft_orig1 = compute_fft_log(original1)
    fft_orig2 = compute_fft_log(original2)
    fft_low = compute_fft_log(low_pass)
    fft_high = compute_fft_log(high_pass)
    fft_hybrid = compute_fft_log(hybrid)
    
    return fft_orig1, fft_orig2, fft_low, fft_high, fft_hybrid

def gaussian_stack(img, sigma, levels):
    stack = []
    for i in range(levels):
        curr_sigma = sigma * (np.sqrt(2) ** i)
        g = gaussian_filter(img, curr_sigma)
        stack.append(g)
    return stack

def laplacian_stack(img, sigma, levels):
    G = gaussian_stack(img, sigma, levels)
    L = []
    for i in range(levels - 1):
        L.append(G[i] - G[i + 1])
    L.append(G[-1])
    return L

def create_vertical_mask(shape, side='left', feather_width=50):
    h, w = shape[:2]
    mask = np.zeros((h, w))
    x = np.arange(w)   
    if side == 'left':
        center = w // 2
        mask = 0.5 * (1 + np.tanh((center - x) / (feather_width / 6)))
        mask = np.broadcast_to(mask, (h, w))
    else:
        center = w // 2
        mask = 0.5 * (1 + np.tanh((x - center) / (feather_width / 6)))
        mask = np.broadcast_to(mask, (h, w))
    return mask.astype(np.float32)

def multiresolution(apple, orange, mask, sigma, levels):
    L_apple = laplacian_stack(apple, sigma, levels)
    L_orange = laplacian_stack(orange, sigma, levels)
    G_mask = gaussian_stack(mask, sigma, levels)
    
    L_blend = []
    for i in range(levels):
        blended_level = np.zeros_like(L_apple[i])
        for c in range(3):
            blended_level[:, :, c] = G_mask[i] * L_apple[i][:, :, c] + (1 - G_mask[i]) * L_orange[i][:, :, c]
        L_blend.append(blended_level)
    
    result = L_blend[-1].copy()
    for i in range(levels - 2, -1, -1):
        result += L_blend[i]
    
    return np.clip(result, 0, 1), L_blend

def create_smooth_circular_mask(shape, center=(310,1050), radius=225, smoothing=10):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = 0.5 * (1 + np.tanh((radius - distance) / smoothing))
    return mask.astype(np.float32)

def save_output(result, name, grayscale=False):
    if grayscale:
        result = exposure.rescale_intensity(result, in_range="image", out_range=(0, 1))
    im_out_uint8 = skimage.img_as_ubyte(result)
    io.imsave(name, im_out_uint8)

if __name__ == "__main__":
    # read in image
    image = io.imread("face.jpg")
    image = color.rgb2gray(image)

    filter = np.ones((9, 9)) / 81
    result = convolution_four_loops(image, filter)
    save_output(result, "box_convolved_four_face.jpg")
    result2 = convolution_two_loops(image, filter)
    save_output(result2, "box_convolved_two_face.jpg")
    result3 = signal.convolve2d(image, filter, mode='same', boundary='fill', fillvalue=0)
    save_output(result3, "box_convolved_function_face.jpg", grayscale=True)

    dx = np.array([[-1, 0, 1]], dtype=np.float64)
    result = convolution_two_loops(image, dx)
    save_output(result, "dx_convolved_face.jpg", grayscale=True)
    
    dy = np.array([[1],
                   [0],
                   [-1]], dtype=np.float64)
    result = convolution_two_loops(image, dy)
    save_output(result, "dy_convolved_face.jpg", grayscale=True)

    # 1.2
    camera_image = io.imread("cameraman.png")
    camera_image = color.rgb2gray(camera_image[:, :, :3])
    grad_x, grad_y = gradients(camera_image, dx, dy)
    save_output(grad_x, "dx_cameraman.png", grayscale=True)
    save_output(grad_y, "dy_cameraman.png", grayscale=True)
    gradient_magnitude = gradient_mag(grad_x, grad_y) 
    save_output(gradient_magnitude, "grad_mag_cameraman.png", grayscale=True)  
    # threshold images
    save_output(threshold(gradient_magnitude, 0.2), "0.2_cameraman.png")
    save_output(threshold(gradient_magnitude, 0.35), "0.35_cameraman.png")

    # 1.3
    G = create_gaussian(0.5)
    smoothed = signal.convolve2d(camera_image, G, mode="same", boundary="symm")
    x, y = gradients(smoothed, dx, dy)
    save_output(x, "smoothed_dx_cameraman.png", grayscale=True)
    save_output(y, "smoothed_dy_cameraman.png", grayscale=True)
    mag = gradient_mag(x, y)
    save_output(mag, "smoothed_mag_cameraman.png", grayscale=True)
    save_output(threshold(mag, 0.2), "0.2_smoothed_mag_cameraman.png")
    # derivative of gaussian
    DoG_x = signal.convolve2d(G, dx, mode='same', boundary="symm")
    DoG_y = signal.convolve2d(G, dy, mode='same', boundary="symm")
    Ix_dog, Iy_dog = gradients(camera_image, DoG_x, DoG_y)
    save_output(Ix_dog, "DoG_x_cameraman.png", grayscale=True)
    save_output(Iy_dog, "DoG_y_cameraman.png", grayscale=True)
    DoG_mag = gradient_mag(Ix_dog, Iy_dog)
    save_output(DoG_mag, "DoG_mag_cameraman.png", grayscale=True)
    save_output(threshold(DoG_mag, 0.2), "0.2_DoG_mag_cameraman.png")
    save_output(G, "gaussian.png", grayscale=True)
    save_output(DoG_x, "DoG_x.png", grayscale=True)
    save_output(DoG_y, "DoG_y.png", grayscale=True)

    # 2.1 - image sharpening
    taj_image = io.imread("taj.jpg")
    taj_image = taj_image.astype(np.float64) / 255.0
    sharp, blur, high_freq = apply_unsharp_mask(taj_image)

    # additional image
    san_diego_image = io.imread("wayfarer.jpg").astype(np.float64) / 255.0
    save_output(apply_unsharp_mask(san_diego_image)[0], "sharp_san_diego.png")
    
    # 2.2 - hybrid images
    # First load images
    # high sf
    im1 = plt.imread('barbie.jpg.webp')/255.

    # low sf
    im2 = plt.imread('oppenheimer.jpg')/255

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im2, im1)

    sigma1 = 5
    sigma2 = 5
    low_pass, high_pass, hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    save_output(hybrid, "hybrid_barbenheimer.jpg")
    show_frequency_analysis(im1, im2, low_pass, high_pass, hybrid)

    orange = io.imread("orange.jpeg").astype(np.float32) / 255.0
    apple = io.imread("apple.jpeg").astype(np.float32) / 255.0
    multiresolution(apple, orange, create_vertical_mask(), 1, 6)