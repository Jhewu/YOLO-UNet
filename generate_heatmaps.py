from custom_yolo_predictor.custom_detection_predictor import CustomDetectionPredictor


import argparse


def YOLO_inference(): 

    return

def generate_heatmaps(): 
    print("Hello World")

    return


if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Using a pre-trained YOLO Ultralytics model (YOLOv12n) 
    to extract bounding boxes from a BraTS dataset, and generate 
    soft gaussian heatmaps for YOLO-UNet training. Heatmaps
    will be saved as a separate directory (to not modified original
    preprocessed images), and its intensity based on YOLO confidence
    """
    # ---------------------------------------------------

    ### Parse arguments
    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=str,help='directory (root) of BraTS dataset\t[stacked_segmentation]')
    parser.add_argument("--yolo_dir", type=str,help='directory of YOLO Ultralytics model weights\t[yolo_checkpoint/weights/best.pt]')
    parser.add_argument('--out_dir',type=str,help='output directory of the generated heatmaps\t[heatmaps]')
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')
    parser.add_argument('--image_size', type=int, help='image size NxN \t[160]')
    parser.add_argument('--batch_size', type=int, help='batch size for each YOLO inference step (speeds up processing significantly) \t[128]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    ### Assign defaults
    IN_DIR = args.data_dir or "stacked_segmentation"
    YOLO_DIR = args.yolo_dir or "yolo_checkpoint/weights/best.pt"
    OUT_DIR = args.out_dir or "heatmaps"
    DEVICE = args.device or "cuda"
    IMAGE_SIZE = args.image_size or 160
    BATCH_SIZE = args.batch_size or 128
    WORKERS = args.workers or 10

    generate_heatmaps()
