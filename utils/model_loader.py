import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet101', pretrained=False)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def load_model(model_path, num_classes=4):
    model = get_model(num_classes)

    state_dict = torch.load(model_path, map_location="cpu")

    # 🔥 INI KUNCI FIX NYA
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model