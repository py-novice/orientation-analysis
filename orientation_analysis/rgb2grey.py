def rgb2grey(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grey