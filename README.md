# GWCS

GWCS contains three steps: 
1. train an original network.
2. search pruned network by GGS algorithm.
3. the pruned network KD from the original network to improving the accuracy.

# Run

1. Requirements:
	python3.6.5; pytorch 1.1.0; torchvision 0.4.0


2. Datasets:
	Cifar10, Cifarf100, ImageNet
 
3. Steps to run:
	train.py: train an original network
	search.py: search pruned network 
	KD.py: improving the accuracy of pruning network

	python train.py
	python search.py
	python KD.py

    
   