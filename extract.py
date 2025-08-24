from numpy.ma.core import masked

from utils import *
from smart_cropper_v2 import SmartCropperV2  # 如果你保存为 smart_cropper_v2.py
# image_path = "C:/Users/Administrator/Desktop/20250724/JinMenShi_CM/S2_420881/S2_420881_merged.tif"
#     mask_path = "E:/test/影像/荆门市/420881/20250726_shpToRaster/420081_20250726_4.tif"
#     output_image_dir = "E:/test/影像/extract_cm/20250726/420081/output_image_dir"
#     output_mask_dir = "E:/test/影像/extract_cm/20250726/420081/output_mask_dir"
#     output_unlabeled_dir = "E:/test/影像/extract_cm/20250726/420081/output_unlabeled_dir"
#     extract_patches_all(image_path,#处理的遥感影像
#     mask_path,#处理的对应标签tif文件
#     output_image_dir,#裁剪成的（512*12或256*256）遥感影像片段文件地址
#     output_mask_dir,#对应的标签片段文件地址
#     output_unlabeled_dir,#对应未打标签文件地址
#     patch_size=256,#对应的片段大小尺寸
#     stride=128,#滑动
#     NODATA_VAL=0,
#     min_crop_pixels = 2,  # 至少2个耕地像素
#     min_valid_ratio=0.05,# 至少2个耕地像素
#     prefix="patch_"# 至少20%像素不是0，才算有效patch)
def main():


    cropper = SmartCropperV2(
        image_path="G:/20250807/origin_data/extract_data/jinmen_merged2023.tif",
        mask_path='G:/20250818/origin_data/extract_data/label_test1.tif',
        region_map_path='G:/20250807/origin_data/extract_data/jimen_region_2023.tif',  # 新增区域引导图路径
        output_image_dir='G:/20250818/origin_data/image_ data',
        output_mask_dir='G:/20250818/origin_data/mask_data',
        output_region_map_dir = 'G:/20250818/origin_data/region_map_dir',  # 输出区域引导图路径
        output_unlabeled_dir='G:/20250818/origin_data/unlabeled_data',
        patch_size=512,
        stride=128,
        classes=[0, 1, 2, 3],  # 根据你的标签类别设定
        ignore_index=255,
        nodata_val=0,
        min_valid_ratio=0.05,
        min_class_ratio=0.05,
    )

    cropper.extract_patches()

if __name__ == '__main__':
    main()