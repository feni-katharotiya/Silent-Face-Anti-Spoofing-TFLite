import torch
import torch.onnx

from collections import OrderedDict


from src.model_lib.MultiFTNet import MultiFTNet

pth_file = "/home/deepkaneria/Documents/AntiSpoofing/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
onnx_file = "/home/deepkaneria/Documents/AntiSpoofing/Silent-Face-Anti-Spoofing-master/converted_models/onnx/2.7_80x80_MiniFASNetV2.onnx"

# # Load your trained model
model = MultiFTNet(img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5))
# model.load_state_dict(torch.load(pth_file, map_location=torch.device("cpu")), strict=False)

checkpoint = torch.load(pth_file, map_location=torch.device("cpu"))
new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    # print(k)
    # new_key = k.replace("module.", "model.")  # Remove 'module.' from keys
    new_key = k.replace("module.", "model.", 1)
    # new_key_2 = new_key.replace("
    new_state_dict[new_key] = v
# exit()

model.load_state_dict(new_state_dict, strict=False)

model.eval()  # Set to evaluation mode

# Create a dummy input tensor (adjust shape based on your model's input)
dummy_input = torch.randn(1, 3, 80, 80)  # Example for an image classifier

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file, 
    export_params=True,  # Store trained parameters
    opset_version=10,  # ONNX version
    do_constant_folding=True,  # Optimize constants
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Handle dynamic batch size
)

print("ONNX model saved as model.onnx")
