import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import glob

def get_model(num_classes, model_type='mobilenet'):
    """Create model architecture (same as in training script)"""
    if model_type == 'resnet50':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        
    elif model_type == 'resnet50_v2':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        
    elif model_type == 'mobilenet':
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    return model

def load_trained_model(model_path, num_classes, model_type='mobilenet', device='cpu'):
    """Load a trained model for inference"""
    model = get_model(num_classes, model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def preprocess_image(image_path):
    """Load and preprocess image for model input"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    
    return image, image_tensor

def predict_image(model, image_tensor, device, confidence_threshold=0.5):
    """Make predictions on a single image"""
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    # Extract predictions
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Filter by confidence threshold
    keep = scores >= confidence_threshold
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }

def visualize_prediction(image, predictions, class_names, confidence_threshold=0.5, 
                        save_path=None, show_plot=True):
    """Visualize predictions on image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Color map for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, (box, score, label) in enumerate(zip(predictions['boxes'], 
                                               predictions['scores'], 
                                               predictions['labels'])):
        if score >= confidence_threshold:
            # Get class name
            class_name = class_names.get(label, f"Class_{label}")
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            color = colors[label % len(colors)]
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and confidence score
            ax.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                   fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'Mango Detection Results', fontsize=14, weight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def get_image_paths(dataset_dir, max_images=None):
    """Get all image paths from dataset directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        # Search recursively in all subdirectories
        paths = glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True)
        image_paths.extend(paths)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    return sorted(image_paths)

def test_model_interactive(model_path, dataset_dir, num_classes, class_names, 
                         model_type='mobilenet', confidence_threshold=0.5, 
                         max_images=None, save_results=False):
    """Test model on images with interactive viewing"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_trained_model(model_path, num_classes, model_type, device)
    print("Model loaded successfully!")
    
    # Get image paths
    image_paths = get_image_paths(dataset_dir, max_images)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Check your dataset directory path.")
        return
    
    # Create output directory if saving results
    if save_results:
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Test images
    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*50}")
        print(f"Testing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"Full path: {image_path}")
        
        try:
            # Load and preprocess image
            image, image_tensor = preprocess_image(image_path)
            print(f"Image shape: {image.shape}")
            
            # Make prediction
            predictions = predict_image(model, image_tensor, device, confidence_threshold)
            
            # Print prediction results
            print(f"Detections found: {len(predictions['boxes'])}")
            for j, (box, score, label) in enumerate(zip(predictions['boxes'], 
                                                       predictions['scores'], 
                                                       predictions['labels'])):
                class_name = class_names.get(label, f"Class_{label}")
                print(f"  Detection {j+1}: {class_name} (confidence: {score:.3f})")
                print(f"    Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Visualize results
            save_path = None
            if save_results:
                save_path = os.path.join(output_dir, f"result_{i+1:03d}_{os.path.basename(image_path)}")
            
            visualize_prediction(image, predictions, class_names, confidence_threshold, 
                               save_path, show_plot=not save_results)
            
            if not save_results:
                # Interactive mode - wait for user input
                user_input = input("\nPress Enter for next image, 'q' to quit, 's' to save this result: ")
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 's':
                    save_path = f"saved_result_{i+1:03d}_{os.path.basename(image_path)}"
                    visualize_prediction(image, predictions, class_names, confidence_threshold, 
                                       save_path, show_plot=False)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

def main():
    # Configuration - MODIFY THESE ACCORDING TO YOUR SETUP
    MODEL_PATH = "mango_detection_model_20.pth"  # Path to your trained model
    DATASET_DIR = "dataset-3000"  # Path to your dataset directory
    
    # Model configuration - should match your training setup
    CLASS_NAMES = {
        0: "bruised",  # Background class (usually not shown)
        1: "not_bruised",        # Adjust these based on your actual classes
        2: "yellow",      # You need to check what classes your model was trained on
        3: "green_yellow",      # Add/remove classes as needed
        4: "green",
        5: "mango",
        6: "background"
    }
    
    NUM_CLASSES = len(CLASS_NAMES)  # Should match your training setup
    MODEL_TYPE = 'mobilenet'  # Should match what you used in training ('mobilenet', 'resnet50', 'resnet50_v2')
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to show detections
    
    # Testing options
    MAX_IMAGES = 20  # Set to None to test all images, or a number to limit
    SAVE_RESULTS = False  # Set to True to save all results to disk instead of interactive viewing
    
    print("Mango Detection Model Tester")
    print("=" * 40)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check the path and try again.")
        return
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        print("Please check the path and try again.")
        return
    
    # Run testing
    test_model_interactive(
        model_path=MODEL_PATH,
        dataset_dir=DATASET_DIR,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        model_type=MODEL_TYPE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_images=MAX_IMAGES,
        save_results=SAVE_RESULTS
    )

if __name__ == "__main__":
    main()