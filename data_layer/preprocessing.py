# get the upper Y coordinate of the ROI
def getH1(image, width, height):
    for i in range(1, height, 1):
        for j in range(1, width, 1):
            pixel = image.getpixel((j, i))
            if pixel > 100:
                return i-1


# get the lower Y coordinate of the ROI
def getH2(image, width, height):
    for i in range(height-1, 1, -1):
        for j in range(width-1, 1, -1):
            pixel = image.getpixel((j, i))
            if pixel > 50:
                return i+1


# crop ROI (Region of Interest)
def crop_retina_image(image, width, height):
    h1 = getH1(image, width, height)
    h2 = getH2(image, width, height)

    image = image.crop((0, h1, width, h2))
    return image
