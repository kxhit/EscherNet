import numpy as np
import einops
import torch
from scipy.spatial.transform import Rotation as R

########################## 6DoF CaPE ####################################
class CaPE_6DoF:
    def cape_embed(self, f, P):
        """
        Apply CaPE on feature.
        :param f: feature vector of shape [..., d]
        :param P: 4x4 transformation matrix
        :return: rotated feature f by pose P: f@P
        """
        f = einops.rearrange(f, '... (d k) -> ... d k', k=4)
        return einops.rearrange(f@P, '... d k -> ... (d k)', k=4)

    def attn_with_CaPE(self, f1, f2, p1, p2):
        """
        Do attention dot production with CaPE pose encoding.
        # query = cape_embed(query, p_out_inv)  # query f_q @ (p_out)^(-T)
        # key = cape_embed(key, p_in)  # key f_k @ p_in
        :param f1: b (t1 l) d
        :param f2: b (t2 l) d
        :param p1: [b, t, 4, 4]
        :param p2: [b, t, 4, 4]
        :return: attention score: q@k.T
        """
        l = f1.shape[1] // p1.shape[1]
        assert f1.shape[1] // p1.shape[1] == f2.shape[1] // p2.shape[1]
        p1_invT = einops.repeat(torch.inverse(p1).permute(0, 1, 3, 2), 'b t m n -> b (t l) m n', l=l)  # f1 [b, l*t1, d]
        query = self.cape_embed(f1, p1_invT)  # [b, l*t1, d] query: f1 @ (p1)^(-T), transpose the last two dim
        p2_copy = einops.repeat(p2, 'b t m n -> b (t l) m n', l=l) # f2 [b, l*t2, d]
        key = self.cape_embed(f2, p2_copy)  # [b, l*t2, d] key: f2 @ p2
        att = query @ key.permute(0, 2, 1)  # [b, l*t1, l*t2] attention: query@key^T
        return att


################### 6DoF Verification ###################################
def euler_to_matrix(alpha, beta, gamma, x, y, z):
    # radian
    r = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)
    t = np.array([[x], [y], [z]])
    rot_matrix = r.as_matrix()
    rot_matrix = np.concatenate([rot_matrix, t], axis=-1)
    rot_matrix = np.concatenate([rot_matrix, [[0, 0, 0, 1]]], axis=0)
    return rot_matrix

def random_6dof_pose(B, T):
    pose_euler = torch.rand([B, T, 6]).numpy()  # euler
    pose_matrix = []
    for b in range(B):
        p = []
        for t in range(T):
            p.append(torch.from_numpy(euler_to_matrix(*pose_euler[b, t])))
        pose_matrix.append(torch.stack(p))
    pose_matrix = torch.stack(pose_matrix)
    return pose_matrix.float()

bs = 6  # batch size
t1 = 3  # num of target views in each batch, can be arbitrary number
t2 = 5  # num of reference views in each batch, can be arbitrary number
l = 10  # len of token
d = 16  # dim of token feature, need to mod 4 in this case
assert d % 4 == 0

# random init query and key
f1 = torch.rand(bs, t1, l, d)     # query
f2 = torch.rand(bs, t2, l, d)     # key
f1 = einops.rearrange(f1, 'b t l d -> b (t l) d')
f2 = einops.rearrange(f2, 'b t l d -> b (t l) d')

# random init pose p1, p2, delta_p, [bs, t, 4, 4]
p1 = random_6dof_pose(bs, t1)   # [bs, t1, 4, 4]
p2 = random_6dof_pose(bs, t2)   # [bs, t2, 4, 4]
p_delta = random_6dof_pose(bs, 1)   # [bs, 1, 4, 4]
# delta p is identical to p1 and p2 in each batch
p1_delta = einops.repeat(p_delta, 'b 1 m n -> b (1 t) m n', t=t1//1)
p2_delta = einops.repeat(p_delta, 'b 1 m n -> b (1 t) m n', t=t2//1)

# run attention with CaPE 6DoF
cape_6dof = CaPE_6DoF()
# att
att = cape_6dof.attn_with_CaPE(f1, f2, p1, p2)
# att_delta
att_delta = cape_6dof.attn_with_CaPE(f1, f2, p1@p1_delta, p2@p2_delta)

# condition: att score should be the same i.e. non effect from any delta_p
assert torch.allclose(att, att_delta, 1e-3)
print("6DoF CaPE Verified")




########################## 4DoF CaPE ####################################
class CaPE_4DoF:
    def rotate_every_two(self, x):
        x = einops.rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return einops.rearrange(x, '... d j -> ... (d j)')

    def cape(self, x, p):
        d, l, n = x.shape[-1], p.shape[-2], p.shape[-1]
        assert d % (2 * n) == 0
        m = einops.repeat(p, 'b l n -> b l (n k)', k=d // n)
        return m

    def cape_embed(self, qq, kk, p1, p2):
        """
        Embed camera position encoding into attention map
        :param qq: query feature map   [b, l_q, feature_dim]
        :param kk: key feature map    [b, l_k, feature_dim]
        :param p1: query pose  [b, l_q, pose_dim]
        :param p2: key pose    [b, l_k, pose_dim]
        :return: cape embedded attention map    [b, l_q, l_k]
        """
        assert p1.shape[-1] == p2.shape[-1]
        assert qq.shape[-1] == kk.shape[-1]
        assert p1.shape[0] == p2.shape[0] == qq.shape[0] == kk.shape[0]
        assert p1.shape[1] == qq.shape[1]
        assert p2.shape[1] == kk.shape[1]

        m1 = self.cape(qq, p1)
        m2 = self.cape(kk, p2)

        q = (qq * m1.cos()) + (self.rotate_every_two(qq) * m1.sin())
        k = (kk * m2.cos()) + (self.rotate_every_two(kk) * m2.sin())

        return q, k

    def attn_with_CaPE(self, f1, f2, p1, p2):
        """
        Do attention dot production with CaPE pose encoding.
        # query = cape_embed(query, p_out_inv)  # query f_q @ (p_out)^(-T)
        # key = cape_embed(key, p_in)  # key f_k @ p_in
        :param f1: b (t1 l) d
        :param f2: b (t2 l) d
        :param p1: [b, t, 4]
        :param p2: [b, t, 4]
        :return: attention score: q@k.T
        """
        l = f1.shape[1] // p1.shape[1]
        assert f1.shape[1] // p1.shape[1] == f2.shape[1] // p2.shape[1]
        p1_reshape = einops.repeat(p1, 'b t m -> b (t l) m', l=l)  # f1 [b, l*t1, d]
        p2_reshape = einops.repeat(p2, 'b t m -> b (t l) m', l=l)  # f1 [b, l*t1, d]
        query, key = self.cape_embed(f1, f2, p1_reshape, p2_reshape)
        att = query @ key.permute(0, 2, 1)  # [b, l*t1, l*t2] attention: query@key^T
        return att

################### 4DoF Verification ###################################
def random_4dof_pose(B, T):
    pose = torch.zeros([B, T, 4])
    pose[:, :, :3] = torch.rand([B, T, 3])  # radian angle
    # theta \in [0, pi], azimuth  \in [0, 2pi], radius \in [0, pi], 0
    pose[:, :, 1] *= (2*torch.pi)
    pose[:, :, 0] *= torch.pi
    pose[:, :, 2] *= torch.pi
    return pose.float()

def look_at(origin, target, up):
    forward = (target - origin)
    forward = forward / torch.linalg.norm(forward, dim=-1, keepdim=True)
    right = torch.linalg.cross(forward, up)
    right = right / torch.linalg.norm(right, dim=-1, keepdim=True)
    new_up = torch.linalg.cross(forward, right)
    new_up = new_up / torch.linalg.norm(new_up, dim=-1, keepdim=True)
    rotation_matrix = torch.stack((right, new_up, forward, target), dim=-1)
    matrix = torch.cat([rotation_matrix, torch.tensor([[0, 0, 0, 1]]).repeat(rotation_matrix.shape[0],rotation_matrix.shape[1], 1, 1)], dim=-2)

    return matrix

def pose_4dof2matrix(pose_4dof):
    """

    :param pose_4dof: [b, t, 4]
    :return: pose 4x4 matrix: [b, t, 4, 4]
    """
    theta = pose_4dof[:, :, 0]
    azimuth = pose_4dof[:, :, 1]
    radius = pose_4dof[:, :, 2]
    xyz = torch.stack([torch.sin(theta) * torch.cos(azimuth), torch.sin(theta) * torch.sin(azimuth), torch.cos(theta)], dim=-1) * radius.unsqueeze(-1)
    origin = torch.zeros_like(xyz)
    up = torch.zeros_like(xyz)
    up[:, :, 2] = 1
    pose = look_at(origin, xyz, up)
    return pose

def pose_matrix24dof(pose_matrix):
    """

    :param pose_matrix: [b, t, 4, 4]
    :return: pose_4dof: [b, t, 4]   theta, azimuth, radius, 0, looking at origin
    """
    xyz = pose_matrix[..., :3, 3]
    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    radius = torch.sqrt(xy + xyz[..., 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[..., 1], xyz[..., 0])
    pose = torch.stack([theta, azimuth, radius, torch.zeros_like(radius)], dim=-1)
    # move to [0, 2pi]
    pose %= (2 * torch.pi)

    return pose

bs = 6  # batch size
t1 = 3  # num of target views in each batch, can be arbitrary number
t2 = 5  # num of reference views in each batch, can be arbitrary number
l = 10  # len of token
d = 16  # dim of token feature, need to mod 4 in this case

# random init query and key
f1 = torch.rand(bs, t1, l, d)     # query
f2 = torch.rand(bs, t2, l, d)     # key
f1 = einops.rearrange(f1, 'b t l d -> b (t l) d')
f2 = einops.rearrange(f2, 'b t l d -> b (t l) d')

#random init 4DoF pose [bs, t1, 4], theta, azimuth, radius, 0
p1 = random_4dof_pose(bs, t1)   # [bs, t1, 4]
p2 = random_4dof_pose(bs, t2)   # [bs, t2, 4]
p1_matrix = pose_4dof2matrix(p1)
p1_4dof = pose_matrix24dof(p1_matrix)
assert torch.allclose(p1, p1_4dof)

p_delta_4dof = random_4dof_pose(bs, 1)
# delta p is identical to p1 and p2 in each batch
p1_delta_4dof = einops.repeat(p_delta_4dof, 'b 1 m -> b (1 t) m', t=t1//1)
p2_delta_4dof = einops.repeat(p_delta_4dof, 'b 1 m -> b (1 t) m', t=t2//1)

# run attention with CaPE 6DoF
cape_4dof = CaPE_4DoF()
# att
att = cape_4dof.attn_with_CaPE(f1, f2, p1, p2)
# att_delta
att_delta = cape_4dof.attn_with_CaPE(f1, f2, p1+p1_delta_4dof, p2+p2_delta_4dof)

# condition: att score should be the same i.e. non effect from any delta_p
assert torch.allclose(att, att_delta, 1e-3)
print("4DoF CaPE Verified")

# print("You should get assertion error because 4DoF CaPE cannot handle 6DoF jitter")
# # att_delta_6dof, it cannot handle 6dof jitter
# p_delta_6dof = random_6dof_pose(bs, 1)   # [bs, 1, 4, 4] any delta transformation in 6DoF
# # delta p is identical to p1 and p2 in each batch
# p1_delta_6dof = einops.repeat(p_delta_6dof, 'b 1 m n -> b (1 t) m n', t=t1//1)
# p2_delta_6dof = einops.repeat(p_delta_6dof, 'b 1 m n -> b (1 t) m n', t=t2//1)
# # 4dof pose to 4x4 matrix
# p1_matrix = pose_4dof2matrix(p1)
# p2_matrix = pose_4dof2matrix(p2)
# att_delta_6dof = cape_4dof.attn_with_CaPE(f1, f2, pose_matrix24dof(p1_matrix@p1_delta_6dof), pose_matrix24dof(p2_matrix@p2_delta_6dof))
# # condition: att score should be the same i.e. non effect from any delta_p
# assert torch.allclose(att, att_delta_6dof, 1e-3)
