def find_image_shift(
                    img1: np.ndarray, 
                    img2: np.ndarray,
                    wavelet='db1', 
                    level=2):
    # Perform wavelet decomposition on the two images
    coeffs1 = pywt.wavedec2(img1, wavelet, level=level)
    coeffs2 = pywt.wavedec2(img2, wavelet, level=level)

    # Compute the correlation between the wavelet coefficients
    correlation = []
    for i in range(len(coeffs1)):
        if isinstance(coeffs1[i], tuple):
            continue  # skip the LL band
        corr = np.fft.ifft2(np.fft.fft2(coeffs1[i]) * np.fft.fft2(coeffs2[i]).conj()).real
        norm = np.linalg.norm(coeffs1[i]) * np.linalg.norm(coeffs2[i])
        correlation.append(corr / norm)

    # Find the shift from the center of the correlation to the max correlation value
    max_idx = np.unravel_index(np.argmax(correlation[0]), correlation[0].shape)
    center = np.array(correlation[0].shape) // 2
    shift = tuple(max_idx - center)

    return shift
