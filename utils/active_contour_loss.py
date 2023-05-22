import torch
import torch.nn as nn
import numpy as np
# https://github.com/HiLab-git/ACELoss/blob/main/aceloss.py

class ACLoss(nn.Module):
    """
    Active Contour Loss
    based on sobel filter
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLoss, self).__init__()

        self.miu = miu
        self.classes = classes
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.sobel_x = nn.Parameter(torch.from_numpy(sobel_x).float().expand(
            self.classes, 1, 3, 3),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.from_numpy(sobel_y).float().expand(
            self.classes, 1, 3, 3),
                                    requires_grad=False)

        self.diff_x = nn.Conv2d(self.classes,
                                self.classes,
                                groups=self.classes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.diff_x.weight = self.sobel_x
        self.diff_y = nn.Conv2d(self.classes,
                                self.classes,
                                groups=self.classes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.diff_y.weight = self.sobel_y

    def forward(self, predication, label):
        grd_x = self.diff_x(predication)
        grd_y = self.diff_y(predication)

        # length
        length = torch.sum(torch.abs(torch.sqrt(grd_x**2 + grd_y**2 + 1e-8)))
        length = (length - length.min()) / (length.max() - length.min() + 1e-8)
        length = torch.sum(length)
        # region_
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in)**2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out)**2)))
        region = self.miu * region_in + region_out

        return region + length


class ACLossV2(nn.Module):
    """
    Active Contour Loss
    based on maxpooling & minpooling
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLossV2, self).__init__()

        self.miu = miu
        self.classes = classes

    def forward(self, predication, label):
        min_pool_x = nn.functional.max_pool2d(predication * -1,
                                              (3, 3), 1, 1) * -1
        contour = torch.relu(
            nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)

        # length
        length = torch.sum(torch.abs(contour))

        # region_
        label = label.float()
        c_in = torch.ones_like(predication)
        c_out = torch.zeros_like(predication)
        region_in = torch.abs(torch.sum(predication * ((label - c_in)**2)))
        region_out = torch.abs(
            torch.sum((1 - predication) * ((label - c_out)**2)))
        region = self.miu * region_in + region_out

        return region + length


def ACELoss(y_pred, y_true, u=1, a=1, b=1):
    """
    Active Contour Loss
    based on total variations and mean curvature
    """

    def first_derivative(input):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        ci_0 = (u[:, :, 1, :] - u[:, :, 0, :]).unsqueeze(2)
        ci_1 = u[:, :, 2:, :] - u[:, :, 0:m - 2, :]
        ci_2 = (u[:, :, -1, :] - u[:, :, m - 2, :]).unsqueeze(2)
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

        cj_0 = (u[:, :, :, 1] - u[:, :, :, 0]).unsqueeze(3)
        cj_1 = u[:, :, :, 2:] - u[:, :, :, 0:n - 2]
        cj_2 = (u[:, :, :, -1] - u[:, :, :, n - 2]).unsqueeze(3)
        cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

        return ci, cj

    def second_derivative(input, ci, cj):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        cii_0 = (u[:, :, 1, :] + u[:, :, 0, :] -
                 2 * u[:, :, 0, :]).unsqueeze(2)
        cii_1 = u[:, :, 2:, :] + u[:, :, :-2, :] - 2 * u[:, :, 1:-1, :]
        cii_2 = (u[:, :, -1, :] + u[:, :, -2, :] -
                 2 * u[:, :, -1, :]).unsqueeze(2)
        cii = torch.cat([cii_0, cii_1, cii_2], 2)

        cjj_0 = (u[:, :, :, 1] + u[:, :, :, 0] -
                 2 * u[:, :, :, 0]).unsqueeze(3)
        cjj_1 = u[:, :, :, 2:] + u[:, :, :, :-2] - 2 * u[:, :, :, 1:-1]
        cjj_2 = (u[:, :, :, -1] + u[:, :, :, -2] -
                 2 * u[:, :, :, -1]).unsqueeze(3)

        cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

        cij_0 = ci[:, :, :, 1:n]
        cij_1 = ci[:, :, :, -1].unsqueeze(3)

        cij_a = torch.cat([cij_0, cij_1], 3)
        cij_2 = ci[:, :, :, 0].unsqueeze(3)
        cij_3 = ci[:, :, :, 0:n - 1]
        cij_b = torch.cat([cij_2, cij_3], 3)
        cij = cij_a - cij_b

        return cii, cjj, cij

    def region(y_pred, y_true, u=1):
        label = y_true.float()
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        region_in = torch.abs(torch.sum(y_pred * ((label - c_in)**2)))
        region_out = torch.abs(torch.sum((1 - y_pred) * ((label - c_out)**2)))
        region = u * region_in + region_out
        return region

    def elastica(input, a=1, b=1):
        ci, cj = first_derivative(input)
        cii, cjj, cij = second_derivative(input, ci, cj)
        beta = 1e-8
        length = torch.sqrt(beta + ci**2 + cj**2)
        curvature = (beta + ci ** 2) * cjj + \
                    (beta + cj ** 2) * cii - 2 * ci * cj * cij
        curvature = torch.abs(curvature) / ((ci**2 + cj**2)**1.5 + beta)
        elastica = torch.sum((a + b * (curvature**2)) * torch.abs(length))
        return elastica

    loss = region(y_pred, y_true, u=u) + elastica(y_pred, a=a, b=b)
    return loss

