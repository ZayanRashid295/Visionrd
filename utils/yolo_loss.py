import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, img_size=416):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.img_size = img_size
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
    def forward(self, predictions, targets):
        """
        predictions: [batch_size, 3, H, W, 5+num_classes]
        targets: List of target dictionaries containing 'boxes' and 'labels'
        """
        batch_size = predictions.size(0)
        loss = torch.tensor(0.0, device=predictions.device)
        
        # Split predictions
        pred_boxes = predictions[..., :4]   # [batch, 3, H, W, 4]
        pred_conf = predictions[..., 4]     # [batch, 3, H, W]
        pred_cls = predictions[..., 5:]     # [batch, 3, H, W, num_classes]
        
        # Calculate losses
        box_loss = self.mse_loss(pred_boxes, targets['boxes'])
        conf_loss = self.bce_loss(pred_conf, targets['conf'])
        cls_loss = self.bce_loss(pred_cls, targets['class'])
        
        loss = box_loss + conf_loss + cls_loss
        return loss / batch_size

# Example usage
anchors = [(10, 13), (16, 30), (33, 23)]
img_size = 416
yolo_loss = YOLOLoss(num_classes=11, anchors=anchors, img_size=img_size)