import torch
import numpy as np
class LArTPC_general():
    def __init__(self, cfg: dict):
        self.lx = cfg.get('length', 6)  # horizontal length as viewed from beam dir
        self.ly = cfg.get('width', 6)  # vertical length as viewed from beam dir
        self.lz = cfg.get('height', 6)  # depth as viewed from beam dir

        self.n_opticsensor = cfg.get('n_opticsensor', 100)
        self.k = 1 / 1.095 # attenuation length in m

        # self.opticsensor = op.

    def place_hexa_opticsensors(self):
        # Calculate spacing based on the number of dots and cube size
        grid_y = int(np.sqrt(self.n_opticsensor) * (self.ly / self.lz))
        grid_z = int(np.sqrt(self.n_opticsensor) / (self.ly / self.lz))

        print(f"Total PMT number is {2*grid_y*grid_z}")

        spacing_y = self.ly / grid_y
        spacing_z = self.lz / grid_z

        # Generate hexagonal grid coordinates
        y_side, z_side = torch.meshgrid(torch.arange(grid_y), torch.arange(grid_z))
        y_side = y_side * spacing_y - self.ly / 2
        z_side = z_side * spacing_z - self.lz / 2
        y_side = y_side.to(torch.float32)
        z_side = z_side.to(torch.float32)
        for i in range(y_side.shape[1]):
            if i % 2 == 1:
                y_side[:,i] += spacing_y / 2

        y = torch.tile(y_side, (2, 1, 1)).reshape(2, -1)
        z = torch.tile(z_side, (2, 1, 1)).reshape(2, -1)
        x = torch.vstack((torch.full((y.shape[1],), self.lx/2), torch.full((y.shape[1],), -self.lx/2)))
        #print(x, y, z)
        # Shift every other row to create a hexagonal pattern

        return torch.column_stack((x.flatten(), y.flatten(), z.flatten()))

    def geometric_factor(self, coords, pmt_pos):
        assert coords.shape[1] == pmt_pos.shape[1], ValueError("Position coordinates not correct.")
        r = torch.cdist(coords, pmt_pos)
        mask = r > 0
        displace = coords[:, None, 0] - pmt_pos[None,:,0]
        sin_angle = displace / r
        #visi_factor = torch.zeros_like(r)
        #factor[mask] = dx[mask].abs() * torch.exp(-self.k * r[mask]) / r[mask] ** 2
        visi_factor = torch.exp(-self.k * r) / r ** 2
        return visi_factor*mask, torch.rad2deg(torch.arcsin(sin_angle))*mask, r