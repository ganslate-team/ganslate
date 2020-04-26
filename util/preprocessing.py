def normalize_from_hu(image, MIN_B=-1024.0, MAX_B=3072.0):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return 2*image - 1

def denormalize_to_hu(image, MIN_B=-1024.0, MAX_B=3072.0):
    pass