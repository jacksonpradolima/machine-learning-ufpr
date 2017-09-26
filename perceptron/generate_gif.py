#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================
Generate gif
=====================

Made with Python3. 

The plots generated for each epoch are organized in a gif called final.gif.

Imports used: imageio and os

Required Packages:
	pip3 install imageio
"""
print(__doc__)

import imageio, os

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "GPL"
__version__ = "1.0"

def create_gif(filenames):
	"""
	Generates a gif from a list of images

	Parameters
	------------
	filenames: string
		File names
	"""
	images = []
	for filename in filenames:
		images.append(imageio.imread(filename))
	imageio.mimsave("final.gif", images, duration=0.6)
	
	print("Gif generated! You can see it in plots dir.")

def main():
	print("========================================================")
	
	basedir = os.getcwd() + "/plots"

	if not os.path.isdir("plots"):
		print("There is not a folder called plots")
		return

	# Junto todas as épocas em um único gif	
	os.chdir(basedir)
	sortpngs = sorted([pl for pl in os.listdir(basedir) if pl.endswith('png')], key=lambda a:int(a.split('_')[1].split('.png')[0]))
	create_gif(sortpngs)

	print("\nFinished!")
if __name__ == '__main__':
	main()