from torch.utils.data import DataLoader

from CTHnet import CTHNet
from config import num_classes
from dataset import CroplandInferenceDataset
from utils import  *

def label_and_save_with_model(
    model_path,
    input_dir,
    output_region_map_dir,
    output_dir,
    # model_class,
    num_classes,
    device=None,
    image_suffix=(".tif", ".tiff", ".png")
):
    # 创建数据集和数据加载器
    inference_dataset = CroplandInferenceDataset(input_dir, output_region_map_dir)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = CTHNet(num_classes=num_classes)
    bn_to_gn(model)

    ckpt = torch.load(model_path, map_location="cpu")  # 或 "cuda:0"

    # 1) 取出真正的权重字典
    state = ckpt["model_state_dict"]  # 关键！不要把 ckpt 直接传给 load_state_dict

    model.load_state_dict(state)
    # 1) 取出真正的权重字典
    model.to(device).eval()

    # 推理过程
    with torch.no_grad():
        for images, region_ids,image_name in inference_loader:
            # 将数据传入GPU（如果使用GPU）
            images = images.cuda()
            region_ids = region_ids.cuda()

            # 推理
            output = model(images, region_ids)["main"]  # 返回的output包含模型的预测结果
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            # 保存为 GeoTIFF
            image_name = image_name[0]
            in_path = os.path.join(input_dir, image_name)
            out_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".tif")
            save_prediction_tif(pred, in_path, out_path)
    print(f"\n✅ 所有样本已打标签并保存到：{output_dir}")

# model_path ="G:/20250807/Teacher_train"
# input_dir = "G:/20250807//unlabeled_data"
# output_dir = "G:/20250807//infer_test"
# output_region_map_dir = 'G:/20250807/origin_data/region_map_dir'  # 输出区域引导图路径
# model_class = CTHNet
# label_and_save_with_model(
#     model_path,
#     input_dir,
#     output_region_map_dir,
#     output_dir,
#     model_class,
#     num_classes,
#     device=None,
#     image_suffix=(".tif", ".tiff", ".png")
# )