# A test for our 824 by 824 images
x = np.linspace(-412, 412, 824)
y = np.linspace(-412, 412, 824)
distx, disty = np.meshgrid(x, y)
z = np.sqrt(distx**2 + disty**2)

average = []
for i in range(1, 412):
    m = z < (i + 0.5)
    n = z > (i - 0.5) 
    m = m * n
    avg = np.sum((m * np.abs(np.fft.fftshift(np.fft.fft2(fl_normalized[0]))))) / np.sum(m)
    average.append(avg)

def radial_spec(image: np.array):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    R = np.sqrt(x ** 2 + y ** 2)
    f = lambda r: image[(R >= r - .5) & (R < r + .5)].mean()
    r = np.linspace(0, 824, num = 824)
    mean = np.vectorize(f)(r)
    return r, mean
 

#TEST:
"""
for i in range(5):
    image1 = np.abs(((np.fft.fftn(ib_normalized[i]))*bp_mask((824,824), 200, 15, 5)))
    image2 = np.abs(((np.fft.fftn(fl_normalized[i]))*bp_mask((824,824), 200, 15, 5)))

    fig, ax = plt.subplots(1, 2, figsize=(15,15))
    r1, mean1 = radial_spec(image1)
    r2, mean2 = radial_spec(image2)
    ax[0].plot(r1, mean1)
    ax[1].plot(r2, mean2)
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    ax[0].set_xlim(0,200)
    ax[1].set_xlim(0,200)
    plt.show()
"""
