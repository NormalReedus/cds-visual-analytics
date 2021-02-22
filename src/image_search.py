import os
import cv2
import csv
from pathlib import Path
import argparse

# make sure we only load valid image files
# omitting target_name will just check for file extensions
def valid_image(file, target_name = ''):
	file = file.lower()
	target_name = target_name.lower()

	valid_extensions = (
		'.jpg',
		'.jpeg',
		'.bmp',
		'.png',
		'.webp',
		'.tif',
		'.tiff'
	)

	ext = os.path.splitext(file)[1]
	ext_valid = ext in valid_extensions

	# file is not valid if it is the target image or if it does not have a correct image extension
	return file != target_name and ext_valid

# creates a normalized histogram with all 3 color channels
def color_histogram(image, normalize_function = cv2.NORM_MINMAX):
	hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
	normalized = cv2.normalize(hist, hist, 0, 255, normalize_function)
	return normalized

# main script
def compare_histograms(target_path, collection_dir, outpath, parallel = False):
	# turns dir/image.jpg into...
	target_name = os.path.split(target_path)[1] # image.jpg
	target_basename = os.path.splitext(target_name)[0] # image

	# filter off invalid files to compare with
	collection = [file for file in os.listdir(collection_dir) if valid_image(file, target_name)]

	# just used for printing a progress bar
	if not parallel:
		collection_len = len(collection)
		print(f'There are {collection_len} images to compare with in this collection.')
	
	# create headers for the csv file
	output = [('filename', 'CORREL', 'CHISQR', 'CHISQR_ALT', 'INTERSECT', 'BHATTACHARYYA', 'KL_DIV')]

	# load target image and create histogram
	target_image = cv2.imread(target_path)
	target_hist = color_histogram(target_image)
	
	# indices are just used for the progress bar
	for i, file in enumerate(collection):
		filepath = os.path.join(collection_dir, file)
		
		# load the comparison image and create histogram
		comparison_image = cv2.imread(filepath)
		comparison_hist = color_histogram(comparison_image)
		
		# this is the similarity values
		correl = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_CORREL)
		chisqr = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_CHISQR)
		chisqr_alt = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_CHISQR_ALT)
		intersect = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_INTERSECT)
		bhat = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_BHATTACHARYYA)
		kl_div = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_KL_DIV)
		
		# add to output list
		output.append((file, round(correl, 2), round(chisqr, 2), round(chisqr_alt, 2), round(intersect, 2), round(bhat, 2), round(kl_div, 2)))

		# print the actual progress for every 10%
		if not parallel:
			if i % (collection_len // 10) == 0:
				print(f'{int((i + 1) / (collection_len // 10) * 10)}% done')

	# # Most similar image --- does not work with several comp functions - should be reimplemented with sorting that takes into accoount whether high or low is good
	# if not parallel:
	# 	similar = {
	# 		"name": '',
	# 		"distance": float('inf')
	# 	}

	# 	# loop through all output and keep adding the lowest value to the 'similar' dictionary
	# 	for comparison in output[1:]: # we slice away the headers
	# 		if comparison[1] < similar['distance']:
	# 			similar['name'] = comparison[0]
	# 			similar['distance'] = comparison[1]

	# if not parallel:
	# 	print(f"The image most similar to {target_name} is {similar['name']} with a value of {similar['distance']}")

	# write csv to the given path
	outfile = os.path.join(outpath, f'{target_basename}.csv')
	with open(outfile, 'w', encoding='utf-8') as fh:
		csv.writer(fh).writerows(output)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate histogram comparisons between a target image and a collection of images.')
	parser.add_argument('target_path', help='The path to the image to compare the collection to.')
	parser.add_argument('collection_dir', help='The path to the directory containing the images to compare to the target image.')
	parser.add_argument('outpath', nargs="?", default='./', help='The path to the directory wherein the output comparison data will be saved as a CSV-file named after the target image.')
	args = parser.parse_args()	

	compare_histograms(args.target_path, args.collection_dir, args.outpath)

# Test Command:
# python image_search.py ../data/img/flowers/image_0001.jpg ../data/img/flowers/