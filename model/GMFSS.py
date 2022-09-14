import torch
import torch.nn.functional as F
from model.gmflow.gmflow import GMFlow
from model.MetricNet import MetricNet
from model.FusionNet import AnimeInterp as Fusionnet

class Model:
    def __init__(self):
        self.flownet = GMFlow(num_scales=2, upsample_factor=4) # from gmflow
        self.metricnet = MetricNet() # from SoftmaxSplatting (trained)
        self.fusionnet = Fusionnet() # from wild animation interpolation (trained)

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.fusionnet.eval()

    def device(self,device):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            checkpoint = torch.load('{}/gmflow.pth'.format(path))
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.flownet.load_state_dict(weights, strict=True)
            self.metricnet.load_state_dict(convert(torch.load('{}/metric.pkl'.format(path))))
            self.fusionnet.load_state_dict(convert(torch.load('{}/fusionnet.pkl'.format(path))))

    def inference(self, img0, img1, timesteps=[0.5], scale=1.0, pred_bidir_flow = True):
        img0_l = F.interpolate(img0, scale_factor = scale, mode="bilinear", align_corners=False)
        img1_l = F.interpolate(img1, scale_factor = scale, mode="bilinear", align_corners=False)
        if pred_bidir_flow:
            flow01,flow10 = self.flownet(img0_l, img1_l, pred_bidir_flow=pred_bidir_flow)
        else:
            flow01 = self.flownet(img0_l, img1_l, pred_bidir_flow=pred_bidir_flow)
            flow10 = self.flownet(img1_l, img0_l, pred_bidir_flow=pred_bidir_flow)
        flow01 = F.interpolate(flow01, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
        flow10 = F.interpolate(flow10, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
        
        output = []
        imgs = torch.cat((img0, img1), 1)
        imgs = F.interpolate(imgs,scale_factor=2,mode='bilinear',align_corners=False)
        for timestep in timesteps:
            F0t = timestep * flow01
            F1t = (1 - timestep) * flow10
            metric0, metric1 = self.metricnet(img0, img1, flow01, flow10, mode='net')
            out = self.fusionnet(img0, img1, F0t, F1t, metric0, metric1)
            out = torch.clamp(out,0,1)[0]
            output.append((((out * 255.).byte().cpu().numpy().transpose(1, 2, 0))))

        return output
