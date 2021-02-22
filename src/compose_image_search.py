import os
# import cv2
# import csv
import argparse
from multiprocessing import Pool
from image_search import compare_histograms, valid_image


def main(collection_dir, outpath):

	# all file names in collection
	collection = os.listdir(collection_dir)

	# full path to only the valid images in collection
	files = [os.path.join(collection_dir, file) for file in collection if valid_image(file)]

	# complete set of args for every iteration of compare_histograms
	args = [(file, collection_dir, outpath, True) for file in files] # True for parallel = True

	args = args[:10] # for demoing

	with Pool(os.cpu_count() - 1) as pool:
		pool.starmap(compare_histograms, args) # func to run, iterable to spread out?




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate histogram comparisons between all images in a collection.')
	parser.add_argument('collection_dir', help='The path to the directory containing the images to compare to eachother.')
	parser.add_argument('outpath', nargs="?", default='comparisons', help='The path to the directory wherein the output comparison data will be saved as a CSV-file named after the target image.')
	args = parser.parse_args()	

	main(args.collection_dir, args.outpath)

	# demo command:
	# python compose_image_search.py ../data/img/flowers/