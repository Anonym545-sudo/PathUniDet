import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
# import torchvision.models as models
# from Universal import DLASeg
# from DoDnet import Universal_model
from unet3 import Universal_model

# heads={'seg':15,'kp':4 ,'hm': 4, 'wh': 2, 'reg': 2}
# unet=DLASeg('dla60', heads,
#                  pretrained=True,
#                  down_ratio=2,
#                  head_conv=256).cuda()

unet=Universal_model().cuda()

input=torch.randn(1,3,512,512).cuda()

flops, params = profile(unet, inputs=(input,1,2))

# 将结果转换为更易于阅读的格式
flops, params = clever_format([flops, params], '%.3f')

print(f"FLOPs:{flops}, 参数量：{params}")


# out = unet.forward(input, 1)
