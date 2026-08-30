import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def gaussian_2d(xy, intensity : float, x0 : float, y0 : float, 
                waist_x : float, waist_y : float, theta : float, 
                offset : float):
    """
    2D gaussian function, considering ellipse and tilt.

    I=I_0*Exp(-2r^2/w^2)
    """

    x, y = xy

    a = ( np.cos(theta)**2 / waist_x**2  +  np.sin(theta)**2 / waist_y**2 ) *2
    b = ( np.sin(2*theta)  / waist_x**2  +  np.sin(2*theta)  / waist_y**2 ) * 4
    c = ( np.sin(theta)**2 / waist_x**2  +  np.cos(theta)**2 / waist_y**2 ) * 2

    exponent = a * (x - x0)**2
    exponent += 2 * b * (x - x0) * (y - y0)
    exponent += c * (y - y0)**2

    return intensity * np.exp(-exponent) + offset



def gaussian_2d_fit(image : np.ndarray, show:bool = False):
    """
    Fit the input image with 2d gaussian function.

    Return fit result, covariance and fitted image.
    
    Fit result and covariance containing: [intensity, x0, y0, waist_x, waist_y, theta, offset]

    Fitted image's dimension is enlarged by 20*20.
    """

    image_width = image.shape[0]
    image_height = image.shape[1]
    x = np.linspace(1, image_width, image_width)
    y = np.linspace(1, image_height, image_height)
    xy = np.meshgrid(y, x)
    xy = (xy[0].reshape(-1,), xy[1].reshape(-1,))

    image_reshape = image.reshape(-1,)

    guess_param = [image.max(), image_width/2, image_height/2, 5.5, 5.5, 0, 10]
    param_bounds = ([0, 0, 0, 0, 0, -np.pi/2, -10], 
        [5000, image_width, image_height, image_width, image_height, np.pi/2, 100])

    popt, pcov = opt.curve_fit(gaussian_2d, xy, image_reshape, p0=guess_param, bounds=param_bounds, maxfev=10000)

    xx = np.x = np.linspace(1, image_width, image_width*20)
    yy = np.linspace(1, image_height, image_height*20)
    xxyy = np.meshgrid(xx, yy)
    fitted_image = gaussian_2d(xxyy, *popt)

    if show:
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title('input image')
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.imshow(fitted_image)
        plt.title('fitted image')
        plt.colorbar()

        plt.suptitle('Intensity: {:.0f}, Waist_x/y: {:.2f}/{:.2f}px'.format(popt[0], popt[3], popt[4]))

        plt.tight_layout()

        plt.show()

    return popt, pcov, fitted_image

