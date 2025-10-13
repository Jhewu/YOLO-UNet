# YOLO-UNet: Two-Stage Brain Tumor Segmentation (BraTS-SSA) in Pytorch
This repository contains a two-stage brain tumor segmentation methodology that leverages the strength of YOLO's ability to detect and UNet's ability to segment. The methodology is the following:

	(1) Pretrained YOLOv12 (in BraTS) detects and locate the brain tumors
	(2) A soft gaussian heatmap is generated from the bounding boxes, and stacked as the 5-channel of the multi-modal 4-channel images (e.g., T1c, T1w, T2w, T2f)
	(3) UNet leverages the 5th channel (gaussian heatmap), to guide segmentation

The goal of this methodology, it's an exploration to create an efficient Brain Tumor Segmentation model for the BraTS-SSA (in low resource and compute environment), that also achieves competitive DICE and Hausffdorf score.

## 📁 Structure
```
├── custom_yolo_predictor/ 	# Ultralytics YOLO Custom Predictor (for 4-channel images)
├── custom_yolo_trainer/	# Ulatralytics YOLO Custom Trainer (for 4-channel images)
├── data/  					# Data location
├── modules/    			# YOLO-UNet modules
├── yolo_checkpoint/ 		# Pre-trained YOLOv12 model
├── dataset.py 				# Creates dataset for YOLO-UNet
├── loss.py 				# YOLOU loss
├── predictor.py			# YOLOU predictor
├── trainer.py				# YOLOU trainer
```

## WORK IN PROGRESS
