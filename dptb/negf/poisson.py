from fmm3dpy import lfmm3d
import torch
from dptb.utils.constants import *




class density2Potential(torch.autograd.Function):
    '''
    This solves a poisson equation with dirichlet boundary condition
    '''
    @staticmethod
    def forward(ctx, imgCoord, coord, density, n, dc):
        img_density = density.view(-1,1,1).expand(-1,1,n)
        img_density = torch.cat([img_density, -img_density, img_density, -img_density], dim=1)
        img_density = img_density.reshape(-1)
        V = []
        if coord.requires_grad:
            pgt = 2
        else:
            pgt = 1

        grad_coord = []

        # for i in tqdm(range(density.shape[0]), desc="Calculating Image Charge Summation"):
        #     density_ = torch.cat([density[0:i],density[i+1:]], dim=0)
        #     density_ = torch.cat([density_, img_density], dim=0)
        #     coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)
        #     coord_ = torch.cat([coord_, imgCoord], dim=0)
        #
        #     out = lfmm3d(eps=1e-10, sources=coord_.transpose(1, 0).numpy(), charges=density_.numpy(), dipvec=None,
        #                         targets=coord[i].unsqueeze(1).numpy(), pgt=pgt)
        #
        #     V.append(out.pottarg[0])
        #
        #     if coord.requires_grad:
        #         grad_coord.append(out.gradtarg)
        # ctx.save_for_backward(coord, torch.tensor(grad_coord), imgCoord, torch.tensor(n))
        # return torch.tensor(V) / (4*pi)

        # for i in tqdm(range(density.shape[0]), desc="Calculating Image Charge Summation"):

        for i in range(density.shape[0]):
            density_ = torch.cat([density[0:i],density[i+1:]], dim=0)
            coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)

            out = lfmm3d(eps=1e-10, sources=coord_.transpose(1, 0).numpy(), charges=density_.numpy(), dipvec=None,
                                targets=coord[i].unsqueeze(1).numpy(), pgt=pgt)
            V.append(out.pottarg[0])

            if coord.requires_grad:
                grad_coord.append(out.gradtarg)

        out = lfmm3d(eps=1e-10, sources=imgCoord.transpose(1, 0).numpy(), charges=img_density.numpy(), dipvec=None,
                                targets=coord.transpose(1,0).numpy(), pgt=pgt)
        V = torch.tensor(V) + torch.tensor(out.pottarg)
        if coord.requires_grad:
            grad_coord = torch.tensor(grad_coord) + torch.tensor(out.gradtarg)
        ctx.save_for_backward(coord, torch.tensor(grad_coord), imgCoord, torch.tensor(n), torch.tensor(dc))
        return V / (4*pi*dc)


    @staticmethod
    def backward(ctx, *grad_outputs):
        # to avoid the overflow and overcomplexity, the backward can also be viewed as a fmm.
        coord, grad_coord, imgCoord, n, dc = ctx.saved_tensors

        grad_density = []
        grad_outputs = grad_outputs[0].reshape(-1)
        grad_coord = torch.einsum("i,ijk->jk",grad_outputs, grad_coord)
        grad_coord = grad_coord.transpose(1,0)
        img_grad_outputs = grad_outputs.view(-1, 1, 1).expand(-1, 1, n)
        img_grad_outputs = torch.cat([img_grad_outputs, -img_grad_outputs, img_grad_outputs, -img_grad_outputs])
        img_grad_outputs = img_grad_outputs.reshape(-1)
        for i in range(grad_outputs.shape[0]):
            grad_outputs_ = torch.cat([grad_outputs[0:i],grad_outputs[i+1:]], dim=0)
            coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)

            grad_out = lfmm3d(eps=1e-15, sources=coord_.transpose(1, 0).detach().numpy(), charges=grad_outputs_.detach().numpy(), dipvec=None,
                                targets=coord[i].unsqueeze(1).detach().numpy(), pgt=1)
            grad_density.append(grad_out.pottarg[0])
        grad_out = lfmm3d(eps=1e-15, sources=imgCoord.transpose(1, 0).detach().numpy(), charges=img_grad_outputs.detach().numpy(), dipvec=None,
                                targets=coord.transpose(1,0).detach().numpy(), pgt=1)
        grad_density = torch.tensor(grad_density) + torch.tensor(grad_out.pottarg)

        if len(grad_coord) == 0:
            return None, None, grad_density / (4*pi * dc), None, None
        else:
            return None, grad_coord.squeeze(-1), grad_density / (4*pi*dc), None, None

def getImg(n, coord, d, dim=2):
    zj = coord[:, dim]
    img1 = torch.stack([zj - (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img2 = torch.stack([-zj - 2 * i * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img3 = torch.stack([zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img4 = torch.stack([-zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img = torch.cat([img1, img2, img3, img4], dim=1).view(-1, 4 * n).unsqueeze(2)
    if dim==2:
        xy = coord[:, :2].view(-1, 1, 2).expand(-1, 4 * n, 2)
        xyz = torch.cat((xy,img), dim=2).view(-1, 3)
    elif dim==1:
        x = coord[:, 0].view(-1, 1, 1).expand(-1, 4 * n, 1)
        z = coord[:, 2].view(-1, 1, 1).expand(-1, 4 * n, 1)
        xyz = torch.cat((x,img,z), dim=2).view(-1, 3)
    elif dim==0:
        yz = coord[:, 1:].view(-1, 1, 2).expand(-1, 4 * n, 2)
        xyz = torch.cat((img,yz), dim=2).view(-1, 3)
    else:
        raise ValueError

    return xyz