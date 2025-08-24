import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import glob
import json

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

def batch_test_and_save(model_path, dataset_dir, output_dir, num_classes=7, 
                       model_type='mobilenet', confidence_threshold=0.3, max_images=50):
    """Test model on all images and save results with visualizations"""
    
    # Class names - UPDATE THESE BASED ON YOUR MODEL
    class_names = {
        0: "background",
        1: "ripe",
        2: "unripe", 
        3: "bruised"
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = get_model(num_classes, model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Get all image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        paths = glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True)
        image_paths.extend(paths)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Results storage
    all_results = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Could not load image: {image_path}")
                continue
                
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess for model
            image_tensor = torch.tensor(original_image).permute(2, 0, 1).float() / 255.0
            
            # Make prediction
            with torch.no_grad():
                prediction = model([image_tensor.to(device)])
            
            # Extract results
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            
            # Filter by confidence
            keep = scores >= confidence_threshold
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            # Store results
            image_result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'detections': []
            }
            
            print(f"  Found {len(filtered_boxes)} detections")
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(original_image)
            
            # Draw predictions
            for j, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                class_name = class_names.get(label, f"Class_{label}")
                
                # Store detection info
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'class_id': int(label),
                    'class_name': class_name
                }
                image_result['detections'].append(detection)
                
                # Draw bounding box
                color = colors[label % len(colors)]
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                       fontsize=12, color='white', weight='bold')
                
                print(f"    {class_name}: {score:.3f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Save visualization
            ax.set_title(f'{os.path.basename(image_path)} - {len(filtered_boxes)} detections', 
                        fontsize=14, weight='bold')
            ax.axis('off')
            
            output_filename = f"result_{i+1:03d}_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            output_path = os.path.join(output_dir, "visualizations", output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            all_results.append(image_result)
            
        except Exception as e:
            print(f"  Error processing {image_path}: {str(e)}")
            continue
    
    # Save results summary
    results_summary = {
        'model_path': model_path,
        'dataset_dir': dataset_dir,
        'total_images': len(image_paths),
        'processed_images': len(all_results),
        'confidence_threshold': confidence_threshold,
        'class_names': class_names,
        'results': all_results
    }
    
    # Save as JSON
    json_path = os.path.join(output_dir, "detection_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write("MANGO DETECTION TEST RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"Total images processed: {len(all_results)}\n\n")
        
        # Count detections per class
        class_counts = {}
        total_detections = 0
        for result in all_results:
            for det in result['detections']:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections += 1
        
        f.write(f"Total detections: {total_detections}\n")
        f.write("Detections per class:\n")
        for class_name, count in class_counts.items():
            f.write(f"  {class_name}: {count}\n")
        f.write(f"\nAverage detections per image: {total_detections/len(all_results):.2f}\n")
        
        # Images with most detections
        f.write(f"\nImages with most detections:\n")
        sorted_results = sorted(all_results, key=lambda x: len(x['detections']), reverse=True)
        for i, result in enumerate(sorted_results[:10]):
            f.write(f"  {i+1}. {result['image_name']}: {len(result['detections'])} detections\n")
    
    print(f"\nTesting completed!")
    print(f"Results saved to: {output_dir}")
    print(f"- Visualizations: {os.path.join(output_dir, 'visualizations')}")
    print(f"- JSON results: {json_path}")
    print(f"- Summary report: {summary_path}")
    print(f"Total detections found: {sum(len(r['detections']) for r in all_results)}")

if __name__ == "__main__":
    # CONFIGURATION - UPDATE THESE PATHS
    MODEL_PATH = "mango_detection_model_20.pth"
    DATASET_DIR = "dataset-3000"
    OUTPUT_DIR = "test_results"
    
    # Model settings - should match your training
    NUM_CLASSES = 7  # Adjust based on your model (including background class)
    MODEL_TYPE = 'mobilenet'  # Should match your training setup
    CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to see more detections
    MAX_IMAGES = 50  # Set to None for all images, or limit for testing
    
    print("Starting batch testing...")
    batch_test_and_save(
        model_path=MODEL_PATH,
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        num_classes=NUM_CLASSES,
        model_type=MODEL_TYPE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_images=MAX_IMAGES
    )