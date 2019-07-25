from send_GCR import detect_text
import os 
import cv2
import statistics

state = "ca"

# os.makedirs('usimages/characters_detected_{}'.format(state))
# os.makedirs('usimages/characters_not_detected_{}'.format(state))
# os.makedirs('scraped_plates_{}/characters'.format(state))


# PATH = 'scraped_plates_{}/characters'.format(state)
PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/test/test_write'

# PATH = 'scraped_plates_{}/characters/characters_resized'.format(state)
# PATH = 'usimages/california/characters'
TRACES = os.listdir(PATH)
def save_char_imgs_detected(state):
    for jj, trace in enumerate(TRACES):
        if "png" in trace:
            full_path = PATH+'/'+trace
            detect_text(full_path,jj,state)

save_char_imgs_detected(state)


def find_appr_dim():
    min_rows = 1000
    min_cols = 1000
    max_rows = 0
    max_cols = 0
    sum_cols = 0
    sum_rows = 0
    list_pixel_rows = []
    list_pixel_cols = []
    for jj, trace in enumerate(TRACES):
        if "png" in trace:
            input = cv2.imread(PATH +'/' + trace)
            pixels_row  = input.shape[0]
            pixels_cols = input.shape[1]
            list_pixel_rows.append(pixels_row)
            list_pixel_cols.append(pixels_cols)
            if pixels_cols > max_cols:
                max_cols = pixels_cols
            if pixels_row > max_rows:
                max_rows = pixels_row
            if pixels_cols < min_cols:
                min_cols = pixels_cols
            if pixels_row < min_rows:
                min_rows = pixels_row
            sum_cols = sum_cols + pixels_cols
            sum_rows = sum_rows + pixels_row
    list_pixel_cols.sort()
    list_pixel_rows.sort()
    return statistics.median(list_pixel_rows), statistics.median(list_pixel_cols)
    # print("minimum # rows ", min_rows)
    # print("maximum # rows ", max_rows)
    # print("average rows ", sum_rows/(jj+1))
    # print("median ", statistics.median(list_pixel_rows))
    # print("minimum # cols ", min_cols)
    # print("maximum # cols ", max_cols)
    # print("average cols", sum_cols/(jj+1))
    # print("median ", statistics.median(list_pixel_cols))



