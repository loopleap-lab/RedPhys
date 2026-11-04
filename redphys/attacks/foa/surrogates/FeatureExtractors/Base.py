import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Dict
from kmeans_pytorch import initialize, pairwise_distance
import torch.nn.functional as F
from torchvision import transforms as T
import sys
import os
from contextlib import contextmanager
import numpy as np
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def kmeans_pt(X, k, tol=1e-4, max_iter=1000):
    """kmeans_pytorch's loop (cluster ids, centers); an empty cluster keeps its center (upstream turns it into NaN and never stops) and the loop is capped."""
    X = X.float(); state = initialize(X, k)
    for _ in range(max_iter):
        ids = torch.argmin(pairwise_distance(X, state), dim=1)
        prev = state.clone()
        for j in range(k):
            if (ids == j).any():
                state[j] = X[ids == j].mean(dim=0)
        if torch.sum(torch.sqrt(torch.sum((state - prev) ** 2, dim=1))) ** 2 < tol:
            break
    return ids, state


def cluster_centers(emb, k):
    """kmeans_pytorch centers of emb's rows, assignment without autograd (same values and gradient)."""
    with torch.no_grad():
        ids, _ = kmeans_pt(emb.detach(), k)
    return torch.stack([emb[ids == j].mean(dim=0) for j in range(k) if (ids == j).any()])

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features

class EnsembleFeatureExtractor_ot(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureExtractor_ot, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x)
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img, device):
        return cluster_centers(embedding_img, self.cluster_number)

class EnsembleFeatureExtractor_ot3(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_ot3, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x)
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        return kmeans_pt(embedding_img, 3)[1].to(device)


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss


class EnsembleFeatureLoss_OT(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local=0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            loss_local += self.OT(gt_local, feature_local) * 2

            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)

        loss=loss+loss_local*0.1
        return loss

    def get_cluster_center(self, embedding_img):
        dev = embedding_img.device
        return kmeans_pt(embedding_img, 5)[1].to(dev)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T


class EnsembleFeatureLoss_OT_Auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT_Auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local) * 2
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.1 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K


        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img):
        dev = embedding_img.device
        return kmeans_pt(embedding_img, 5)[1].to(dev)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_foa_attack(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5, eta=0.2, temperature=1.0, sinkhorn_lambda=0.1):
        super(EnsembleFeatureLoss_OT_foa_attack, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.cluster_number = cluster_number
        self.eta = eta
        self.temperature = temperature
        self.sinkhorn_lambda = sinkhorn_lambda


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index]
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)

            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + self.eta * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = (1 + self.eta - total_losses[i].item()) / (1 + self.eta - self.previous_loss_list[i].item() + 1e-8)  # Eq. 12, loss = 1+eta-sim
            weights.append(ratio)

        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / self.temperature)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K


        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img, device):
        return cluster_centers(embedding_img, self.cluster_number)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / self.sinkhorn_lambda)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_ablation_wo_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K


        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        return kmeans_pt(embedding_img, self.cluster_number)[1].to(device)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K


        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        return kmeans_pt(embedding_img, self.cluster_number)[1].to(device)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_dynamic(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_dynamic, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]

        total_loss = sum(total_losses)/len(total_losses)
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        return kmeans_pt(embedding_img, self.cluster_number)[1].to(device)

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T