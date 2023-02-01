def xcorr(
    img1: np.ndarray, 
    img2: np.ndarray,
    hp: float=None,
    lp: float=None,
    sigma: float=None) -> np.ndarray:

    if img1.data.shape != img2.data.shape:
        err = f"Image 1 {img1.data.shape} and Image 2 {img2.data.shape} need to have the same shape"
        logging.error(err)
        raise ValueError(err)

    # Create Fourier transform
    img1fft = np.fft.fftn(img1.data)
    img2fft = np.fft.fftn(img2.data)
    # Do some shady normalization
    n_pixels1 = img1.data.shape[0] * img1.data.shape[1]
    img1fft[0, 0] = 0
    tmp = img1fft * np. conj(img1fft)
    img1fft = n_pixels1 * img1fft / np.sqrt(tmp.sum())
    n_pixels2 = img2.data.shape[0] * img2.data.shape[1]
    img2fft[0, 0] = 0
    tmp = img2fft * np. conj(img2fft)
    img2fft = n_pixels2 * img2fft / np.sqrt(tmp.sum())

    corr_mask = bp_mask(img1.shape, 100, 0, 0)

    if hp and lp and sigma is not None:
        bp = bp_mask(img1.shape, hp, lp, sigma)
    
        # Cross-correlate the two images
        temp = np.real(np.fft.ifftn((img1fft * bp) * np.conj(img2fft)))
        corr = np.fft.fftshift(temp * corr_mask)
        # Cross-correlation center and shift from center
        maxX, maxY = np.unravel_index(np.argmax(corr), corr.shape)
        cen = np.asarray(corr.shape) / 2
        err = np.array(cen - [maxX, maxY], int)
        valMax = np.amax(corr)
        return corr, err

    else:

        # Cross-correlate the two images
        temp = np.real(np.fft.ifftn((img1fft * np.conj(img2fft))))
        corr = np.fft.fftshift(temp * corr_mask)

        # Cross-correlation center and shift from center
        maxX, maxY = np.unravel_index(np.argmax(corr), corr.shape)
        cen = np.asarray(corr.shape) / 2
        err = np.array(cen - [maxX, maxY], int)
        valMax = np.amax(corr)
        return corr, err
