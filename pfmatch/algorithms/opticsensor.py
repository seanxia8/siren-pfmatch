from scipy.special import erf
import torch
import numpy as np

def PMT_PEPC_general(x: torch.Tensor, **kwargs):
    '''
    Calculates the sensor response in pC from a photon
    Parameters
    ----------
    x: tensor of charges in pC, shape (N_pmt, N_charge_ticks)
    Returns
    -------
    profile: tensor of pC probability profile (N_pmt,N_charge_ticks)
    '''

    spe_mean = kwargs.get('spe_mean', 2.1)
    spe_std = kwargs.get('spe_std', 0.5)
    spe_scale = kwargs.get('spe_scale', 5.0)
    erf_mean_low = kwargs.get('erf_mean_low', 0.5)
    erf_mean_high = kwargs.get('erf_mean_high', 2.5)
    erf_std_low = kwargs.get('erf_std_low', 0.2)
    erf_std_high = kwargs.get('erf_std_high', 0.8)
    erf_scale = kwargs.get('erf_scale_low', 0.6)

    gauss = spe_scale * torch.exp(-0.5 * ((x - spe_mean) / spe_std) ** 2) / (spe_std * np.sqrt(2 * np.pi))
    erf_low = erf((x - erf_mean_low) / (np.sqrt(2) * erf_std_low))
    erf_high = erf((x - erf_mean_high) / (np.sqrt(2) * erf_std_high))

    profile = gauss + erf_scale * (erf_low - erf_high)
    profile /= torch.sum(profile, dim=1, keepdim=True) * (x[0, 1] - x[0, 0])
    return profile


def PMT_TTS_general(x: torch.Tensor, **kwargs):
    '''
    Calculates the sensor time response (exponentially modified gaussian) in ns from a photon
    Parameters
    ----------
    x: tensor of times in ns, shape (N_pmt, N_time_ticks)
    Returns
    -------
    profile: tensor of time probability profile (N_pmt,N_time_ticks)
    '''

    tts_mean = kwargs.get('time_mean', 0.0)
    tts_sig = kwargs.get('time_sig', 1.0)
    tts_lambda = kwargs.get('time_lambda', 0.5)

    profile = tts_lambda / 2 * torch.exp(tts_lambda / 2 * (2 * tts_mean + tts_lambda * tts_sig ** 2 - 2 * x)) * \
              (1 - erf((tts_mean + tts_lambda * tts_sig ** 2 - x) / (np.sqrt(2) * tts_sig)))
    profile /= torch.sum(profile, dim=2, keepdim=True) * (x[0, 0, 1] - x[0, 0, 0])
    return profile


def PMT_angle_general(x: torch.Tensor, **kwargs):
    '''
    Calculates the sensor angular efficiency from a photon
    Parameters
    ----------
    x: tensor of angles, shape (N_photon, N_pmt)
    Returns
    -------
    efficiency: tensor of efficiencies (N_photon, N_pmt)
    '''

    sigmoid_coeff = kwargs.get('sigmoid_coeff', 0.15)
    # cos_45 = torch.cos(torch.deg2rad(torch.tensor(45.0)))  # Convert 45 degrees to cosine value
    efficiency = 1 - 1 / (1 + torch.exp(-sigmoid_coeff * (torch.abs(90 - x) - 45)))
    # efficiency = 1/(1+torch.exp(-sigmoid_coeff*(torch.abs(cos_x))))
    return efficiency


def compute_probability_at_x(x: torch.Tensor, profile: torch.Tensor):
    """
    Computes the cumulative integral of the profile at each x (i.e., CDF).

    Parameters:
    x : torch.Tensor
        Charge values, shape (N_pmt, N_charge_ticks)
    profile : torch.Tensor
        Probability density values, shape (N_pmt, N_charge_ticks)

    Returns:
    prob_at_x : torch.Tensor
        Cumulative integral (CDF) of profile at each x, same shape as profile
    """
    dx = x[:, 1:] - x[:, :-1]  # Compute step sizes
    cumulative_integral = torch.cumsum(0.5 * (profile[:, 1:] + profile[:, :-1]) * dx, dim=1)
    # Prepend zero to match original shape (integral starts at 0)
    prob_at_x = torch.cat((torch.zeros((profile.shape[0], 1)), cumulative_integral), dim=1)

    return prob_at_x


class dot_steel_PMT():
    def __init__(self, cfg: dict):
        """
        Calculates the sensor response from a photon at `time_tick` relative to the PE time

        Args:
            cfg (dict): configuration dictionary
        Returns:
            torch.Tensor: response
        """
        self.total_eff = cfg.get('total_eff', 0.1)
        self.pmt_pepc = cfg.get('pmt_pepc', 'general')
        self.pmt_ang_eff = cfg.get('pmt_ang_eff', 'general')
        self.pmt_tts = cfg.get('pmt_tts', 'general')
        self.t_threshold = cfg.get('t_threshold', 0.05)
        self.n_PE = None
        self.pmt_eff = None

        self.charge_max = cfg.get('charge_range', 10)
        self.charge_resolution = cfg.get('charge_resolution', 100)
        self.time_min = cfg.get('time_min', -5)
        self.time_max = cfg.get('time_max', 15)
        self.time_resolution = cfg.get('time_resolution', 100)

    def compute_pmt_eff(self, angle: torch.Tensor, n_photons: torch.Tensor, **kwargs):
        '''
        Computes the PMT efficiency for each photon
        Parameters
        ----------
        angle : angle of the photons seen at each pmt (N_photon, N_pmt)
        n_photons : number of photons reaches each pmt (N_pmt)
        Returns
        -------
        n_PE : number of PE generated at each pmt (N_photon, N_pmt)
        '''
        if self.pmt_ang_eff == 'general':
            self.pmt_eff = PMT_angle_general(angle, **kwargs) * self.total_eff
        else:
            raise ValueError(f"Unknown angular response model: {self.pmt_ang_eff}")

        self.n_PE = n_photons * self.pmt_eff // 1
        self.max_n = int(torch.max(self.n_PE).item())
        self.hit_mask = self.n_PE > 0

    def read_pmt_Q(self, **kwargs):
        """
        Calculates the sensor Q response in pC from the observed photons
        Returns
        -------
        accumulated pmt charge Q
        """
        if self.pmt_pepc == 'general':
            q_axis = torch.linspace(0, self.charge_max, self.charge_resolution).unsqueeze(0).tile(self.n_PE.shape[1], 1)
            profile_q = PMT_PEPC_general(q_axis, **kwargs)
            prob_at_q = compute_probability_at_x(q_axis, profile_q)
        else:
            raise ValueError(f"Unknown pmt_pepc: {self.pmt_pepc}")

        # Generate random numbers between 0 and 1 for each `pe`
        # rand_vals = [[torch.rand(int(n)) for n in row] for row in self.n_PE]
        # Find the maximum length in each row
        rand_vals = torch.rand((self.n_PE.shape[0], self.n_PE.shape[1], self.max_n))
        padded_q = torch.full((self.n_PE.shape[0], self.n_PE.shape[1], self.max_n), 0, dtype=torch.float)

        # Find indices where rand_vals fall in `prob_at_q`
        for i, nPE in enumerate(self.n_PE):
            if self.hit_mask[i].sum() > 1:
                for j, n in enumerate(nPE):
                    if int(n) == 0: continue
                    indices = torch.searchsorted(prob_at_q[j], rand_vals[i, j, :int(n)])
                    padded_q[i, j, :len(indices)] = profile_q[j][indices]
        sum_q = torch.sum(padded_q, dim=2)
        return padded_q, sum_q

    def read_pmt_T(self, mean_time: torch.Tensor, **kwargs):
        """
        Calculates the sensor T response in ns from the observed photons

        Parameters
        ----------
        mean_time: mean time of the photons seen at each pmt by tof (1, N_pmt)
        Returns
        -------
        accumulated pmt time T
        """

        if self.pmt_tts == 'general':
            t_axis = torch.linspace(self.time_min, self.time_max, self.time_resolution).unsqueeze(0).tile(
                self.n_PE.shape[0],
                self.n_PE.shape[1],
                1)
            t_axis += mean_time
            profile_t = PMT_TTS_general(t_axis, time_mean=mean_time)
            prob_at_t = torch.stack([compute_probability_at_x(t_axis[i], profile_t[i]) for i in range(t_axis.shape[0])])
        else:
            raise ValueError(f"Unknown pmt_tts: {self.pmt_tts}")

        # Generate random numbers between 0 and 1 for each `pe`
        rand_vals = torch.rand((self.n_PE.shape[0], self.n_PE.shape[1], self.max_n))
        padded_t = torch.full((self.n_PE.shape[0], self.n_PE.shape[1], self.max_n), -999, dtype=torch.float)
        #mask_t_threshold = prob_at_t > self.t_threshold
        # Find indices where rand_vals fall in `prob_at_t`
        for i, nPE in enumerate(self.n_PE):
            if self.hit_mask[i].sum() > 0:
                for j, n in enumerate(nPE):
                    if int(n) == 0: continue
                    #assert torch.sum(mask_t_threshold[i,j]) > 0, "No time values above threshold"
                    #indices = torch.searchsorted(prob_at_t[i,j][mask_t_threshold[i,j]], rand_vals[i, j, :int(n)])
                    indices = torch.searchsorted(prob_at_t[i, j], rand_vals[i, j, :int(n)])
                    padded_t[i, j, :len(indices)] = profile_t[i, j, indices]

        mask_0 = padded_t > -999
        first_t = torch.min(padded_t * mask_0, dim=2).values
        return padded_t, first_t