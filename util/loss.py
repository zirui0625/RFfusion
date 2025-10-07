import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = L_Grad_position()
        self.loss_func_Max = L_Intensity()
        self.loss_func_color = L_color()
        self.loss_func_mask = L_Mask()

    def forward(self, image_visible, image_infrared, image_fused, max_ratio=4, ssim_vis_ratio=1, ssim_ir_ratio=1, ssim_ratio=1, color_ratio=12, text_ratio=10, mask_ratio=1):
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused)
        loss_ssim = ssim_ratio * (ssim_vis_ratio * self.loss_func_ssim(image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray))
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)
        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        loss_mask = mask_ratio * self.loss_func_mask(image_visible, image_infrared, image_fused)
        total_loss = loss_max + loss_ssim + loss_color + loss_text + loss_mask
        return total_loss, loss_ssim, loss_max, loss_color, loss_text, loss_mask

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        Loss_intensity = F.l1_loss(fused_image, image_fused)
        return Loss_intensity

class L_Grad_position(nn.Module):
    def __init__(self):
        super(L_Grad_position, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)
    
    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class L_Mask(nn.Module):
    def __init__(self, c=10):
        super(L_Mask, self).__init__()
        self.c = c

    def forward(self, image_visible, image_infrared, image_fused):
        B, C, H, W = image_visible.shape
        device = image_visible.device

        losses = []

        for i in range(B):
            vis_np = self.tensor_to_gray_numpy(image_visible[i])  # (H, W)
            ir_np = self.tensor_to_gray_numpy(image_infrared[i])  # (H, W)

            w1, w2 = self.map_generate(ir_np, vis_np, self.c)  # (H, W)

            w1 = torch.tensor(w1, dtype=torch.float32, device=device).unsqueeze(0).expand(C, -1, -1)  # (C, H, W)
            w2 = torch.tensor(w2, dtype=torch.float32, device=device).unsqueeze(0).expand(C, -1, -1)  # (C, H, W)

            fused_masked = w1 * image_visible[i] + w2 * image_infrared[i]  # (C, H, W)

            loss = F.l1_loss(fused_masked, image_fused[i])
            losses.append(loss)

        return torch.stack(losses).mean()

    def map_generate(self, ir_np, vis_np, c=10):
        map1_ = self.vsm(ir_np)
        map2_ = self.vsm(vis_np)
        w1 = 0.4+ (map1_*1-map2_*0.4)
        w2 = 1-w1
        return w1, w2

    def sal(self, his):
        ret = np.zeros(256, int)
        for i in range(256):
            for j in range(256):
                ret[i] += np.abs(j-i)*his[j]
        return ret

    def vsm(self, img):
        img = np.round(img).astype(np.uint8)
        his = np.zeros(256, int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                his[img[i][j]] += 1
        sal = np.zeros(256, int)
        for i in range(256):
            for j in range(256):
                sal[i] += np.abs(j-i)*his[j]
        map = np.zeros_like(img, int)
        for i in range(256):
            map[np.where(img == i)] = sal[i]
        if map.max()==0:
            return np.zeros_like(img, int)
        return map / (map.max())
       
    def tensor_to_gray_numpy(self, img):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 1:
            return img[0]
        elif img.shape[0] == 3:
            r, g, b = img[0], img[1], img[2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray
        else:
            raise ValueError("Unsupported channel number: {}".format(img.shape[0]))



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)