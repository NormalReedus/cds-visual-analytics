import os
import cv2
import csv



def valid_image(file, target_name):
    return file != target_name and (file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'))








def color_histogram(image):
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    normalized = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return normalized






def compare_histograms(target_path, collection_dir, outpath):
    target_name = os.path.split(target_path)[1]

    collection = [file for file in os.listdir(collection_dir) if valid_image(file, target_name)]
    
    output = [('filename', 'distance')]
    
    target_image = cv2.imread(target_path)
    
    target_hist = color_histogram(target_image)
    
    for file in collection:
        filepath = os.path.join(collection_dir, file)
        
        comparison_image = cv2.imread(filepath)
        
        comparison_hist = color_histogram(comparison_image)
        
        chi_square = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_CHISQR)
        
        output.append((file, round(chi_square, 2)))
    
    with open(outpath, 'w', encoding='utf-8') as fh:
        csv.writer(fh).writerows(output)





target = os.path.join('..', 'data', 'img', 'davinci', 'Leonardo_da_Vinci_1.jpg')
collection_dir = os.path.join('..', 'data', 'img', 'davinci')

compare_histograms(target, collection_dir, './vincis.csv')