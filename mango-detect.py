import os, json, cv2, torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
class CVATAnnotationParser:
    
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.labels = self._extract_labels()
        self.annotations = self._extract_annotations()
    
    def _extract_labels(self):
        labels = {}
        label_elements = self.root.findall('.//label')
        for i, label in enumerate(label_elements):
            labels[label.find('name').text] = i
        return labels
    
    def _extract_annotations(self):
        annotations = []
        images = self.root.findall('.//image')
        
        for image in images:
            img_data = {
                'id': int(image.get('id')),
                'filename': image.get('name'),
                'width': int(image.get('width')),
                'height': int(image.get('height')),
                'boxes': []
            }
            
            boxes = image.findall('.//box')
            for box in boxes:
                box_data = {
                    'label': box.get('label'),
                    'label_id': self.labels[box.get('label')],
                    'xtl': float(box.get('xtl')),
                    'ytl': float(box.get('ytl')),
                    'xbr': float(box.get('xbr')),
                    'ybr': float(box.get('ybr'))
                }
                img_data['boxes'].append(box_data)
            
            annotations.append(img_data)
        
        return annotations
    
    def to_yolo_format(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Create classes.txt
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for label in sorted(self.labels.keys()):
                f.write(f"{label}\n")
        
        # Create annotation files
        for ann in self.annotations:
            txt_filename = ann['filename'].replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(output_dir, txt_filename), 'w') as f:
                for box in ann['boxes']:
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (box['xtl'] + box['xbr']) / 2 / ann['width']
                    y_center = (box['ytl'] + box['ybr']) / 2 / ann['height']
                    width = (box['xbr'] - box['xtl']) / ann['width']
                    height = (box['ybr'] - box['ytl']) / ann['height']
                    
                    f.write(f"{box['label_id']} {x_center} {y_center} {width} {height}\n")
    
    def to_coco_format(self):
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Categories
        for label, label_id in self.labels.items():
            coco_data['categories'].append({
                'id': label_id,
                'name': label,
                'supercategory': 'mango'
            })
        
        annotation_id = 0
        for ann in self.annotations:
            # Image info
            coco_data['images'].append({
                'id': ann['id'],
                'file_name': ann['filename'],
                'width': ann['width'],
                'height': ann['height']
            })
            
            # Annotations
            for box in ann['boxes']:
                width = box['xbr'] - box['xtl']
                height = box['ybr'] - box['ytl']
                area = width * height
                
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': ann['id'],
                    'category_id': box['label_id'],
                    'bbox': [box['xtl'], box['ytl'], width, height],
                    'area': area,
                    'iscrowd': 0
                })
                annotation_id += 1
        
        return coco_data

class MangoDataset(Dataset):
    def __init__(self, annotations, image_dir, transforms=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image with error handling
        img_path = os.path.join(self.image_dir, ann['filename'])
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Create a dummy image if file is missing
            image = np.zeros((ann['height'], ann['width'], 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for box in ann['boxes']:
            boxes.append([box['xtl'], box['ytl'], box['xbr'], box['ybr']])
            labels.append(box['label_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return torch.tensor(image).permute(2, 0, 1).float() / 255.0, target

def get_model(num_classes, model_type='resnet50'):
    if model_type == 'resnet50':
        # RECOMMENDED: Best overall performance for mango detection
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        print("Using Faster R-CNN ResNet50-FPN (COCO_V1) - Best overall performance")
        
    elif model_type == 'resnet50_v2':
        # Latest version with improved training protocol
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        print("Using Faster R-CNN ResNet50-FPN V2 - Latest improved version")
        
    elif model_type == 'mobilenet':
        # Lightweight version for mobile/edge deployment
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        print("Using Faster R-CNN MobileNetV3-Large-FPN - Fastest, lowest memory")
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Replace the classifier head for your custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    return model

def train_model(model, data_loader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {losses.item():.4f}')
        
        print(f'Epoch {epoch+1} Average Loss: {epoch_loss/len(data_loader):.4f}')

def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        image, target = dataset[i]
        
        # Make prediction
        with torch.no_grad():
            prediction = model([image.to(device)])
        
        # Convert image back to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'Image {i+1}')
        
        # Plot ground truth boxes (green)
        for box in target['boxes']:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[i].add_patch(rect)
        
        pred_boxes = prediction[0]['boxes'].cpu()
        pred_scores = prediction[0]['scores'].cpu()
        
        for box, score in zip(pred_boxes, pred_scores):
            if score > 0.5:  # Only show confident predictions
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[i].add_patch(rect)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main training pipeline
def main():
    annotations = {'ripe_only':"annotations.xml",
                   'both': "Annotations_color_bruise.xml"}
    parser = CVATAnnotationParser(annotations['both'])
    
    print("Labels found:")
    for label, idx in parser.labels.items():
        print(f"  {idx}: {label}")
    
    print(f"\nTotal images: {len(parser.annotations)}")
    
    # TODO: CHANGE THIS TO YOUR DATASET DIRECTORY
    image_dir = 'cleaned_frames_by_20_og/cleaned_frames_by_20'  
    valid_annotations = []
    missing_files = []
    
    for ann in parser.annotations:
        img_path = os.path.join(image_dir, ann['filename'])
        if os.path.exists(img_path):
            valid_annotations.append(ann)
        else:
            missing_files.append(ann['filename'])
    
    print(f"Valid images found: {len(valid_annotations)}")
    if missing_files:
        print(f"Missing images: {len(missing_files)}")
        print("First 5 missing files:", missing_files[:5])
        
        # Option to continue with valid images only
        response = input("Continue with valid images only? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    parser.annotations = valid_annotations

    parser.to_yolo_format('yolo_annotations')
    coco_data = parser.to_coco_format()
    with open('coco_annotations.json', 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    dataset = MangoDataset(
        parser.annotations, 
        image_dir=image_dir,
        transforms=None
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    num_classes = len(parser.labels) + 1  # +1 for background
    model = get_model(num_classes, 'mobilenet')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.005, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    
    print(f"\nStarting training with {len(valid_annotations)} images...")
    train_model(model, data_loader, optimizer, device, num_epochs=10)
    
    torch.save(model.state_dict(), 'mango_detection_model.pth')
    
    visualize_predictions(model, dataset, device)

if __name__ == "__main__":
    main()

# Additional utility functions for inference
# def load_trained_model(model_path, num_classes):
#     """Load a trained model for inference"""
#     model = get_model(num_classes, 'mobilenet')
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model
#
# def predict_image(model, image_path, device, confidence_threshold=0.5):
#     """Make predictions on a single image"""
#     # Load and preprocess image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
#
#     # Make prediction
#     model.eval()
#     with torch.no_grad():
#         prediction = model([image_tensor.to(device)])
#
#     # Filter predictions by confidence
#     boxes = prediction[0]['boxes'].cpu().numpy()
#     scores = prediction[0]['scores'].cpu().numpy()
#     labels = prediction[0]['labels'].cpu().numpy()
#
#     # Keep only confident predictions
#     keep = scores >= confidence_threshold
#
#     return {
#         'boxes': boxes[keep],
#         'scores': scores[keep],
#         'labels': labels[keep]
#     }
