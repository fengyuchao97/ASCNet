from thop import clever_format
from thop import profile
import torch

from models.TDANet_3D_ALL_GLOBAL import TDANet_3D_ALL_GLOBAL


if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    # model = ICIFNet(pretrained=True)
    # res = res_para()
    # pvt = pvt_para()
    input1 = torch.randn(1, 3, 256, 256).cuda()
    input2 = torch.randn(1, 3, 256, 256).cuda()
    model = TDANet_3D_ALL_GLOBAL().to(device)
    # model = MixAttNet5().to(device)
    # model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8).to(device)
    flops, params = profile(model, inputs=(input1,input2))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
    
    # out1, out2 = model(input1, input2)
    # print(out1.shape)

    # model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8) # CrossNet(pretrained=True)
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops)
    # print(params)

    # input1 = torch.randn(1, 3, 256, 256).cuda()
    # input2 = torch.randn(1, 3, 256, 256).cuda()
    # model = EGRCNN().to(device)
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # # print("CrossNet2:")
    # print(flops)
    # print(params)
    
    # flops, params = profile(res, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("res:")
    # print(flops)
    # print(params)

    # flops, params = profile(pvt, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("pvt:")
    # print(flops)
    # print(params)

    # model = DSIFN()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("IFNet:")
    # print(flops)
    # print(params)

    # model = SNUNet_ECAM()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SNUNet:")
    # print(flops)
    # print(params)

    # model = Unet()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("Unet:")
    # print(flops)
    # print(params)  

    # model = SiamUnet_diff()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SiamUnet_diff:")
    # print(flops)
    # print(params) 

    # model = SiamUnet_conc()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SiamUnet_conc:")
    # print(flops)
    # print(params)  


    # model = CDNet_model(3, SEBasicBlock, [3, 4, 6, 3])
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("DTCDSCN:")
    # print(flops)
    # print(params)     
                     