import numpy as np
from .proposal import ProposalDistribution
from .prior import PriorDistribution
from .likelihood import get_likelihood_ratio

class MetropolisHastings:

    def __init__(self, proposal_width, prior_type, prior_mean, prior_width, likelihood_ratio, infer_std):
        self.proposal_dist = ProposalDistribution(proposal_width)
        self.prior_dist = PriorDistribution(prior_type, prior_mean, prior_width, infer_std)
        self.likelihood_ratio = likelihood_ratio
        self.infer_std = infer_std

    def run(self, N=200000, skip_interval=20, burn_in=5000):
        verbose = False
        if verbose:
            print("Running MCMC:")
        
        if self.infer_std:
            # mean and std
            g = np.array([0, 5])    # start with N(0, 1)
        else:
            g = 0 # initial g

        samples = []    # samples obtained
        all_gs = [g]    # keep track of every g in the chain

        t = 0
        t_skip = 0
        acceptance_count = 0

        while True:

            g_proposal = self.proposal_dist.propose(g)

            prior_ratio = self.prior_dist.prior_ratio(g_proposal, g)
            likelihood_ratio = self.likelihood_ratio(g_proposal, g)
            accept_ratio = prior_ratio * likelihood_ratio
            # print(accept_ratio)

            if np.random.uniform() <= accept_ratio:
                g = g_proposal
                acceptance_count += 1

            if t_skip == 0 and t >= burn_in :
                samples.append(g)
                # print(g)

            all_gs.append(g)
            t_skip = (t_skip + 1) % skip_interval
            t += 1

            if t % 10000 == 0:
                if verbose:
                    print(f"{t} Steps Completed")

            if t == N:
                break

        # if verbose:
        print("Acceptance rate:", acceptance_count / N)

        if self.infer_std:
            # keep only mean
            # print(np.mean(np.array(samples)[:, 1]))
            samples = np.array(samples)[:, 0]

        return samples, all_gs
    

def run_mcmc_default(statistic_name, zhat, datasize, **kwargs):
    """
    Default MCMC settings using jeffrey's pior.

    :ivar infer_std:
        If true, infer std using mcmc.
        Else, use std derived from candidate
        data.
    :vartype infer_std: bool
    """

    # --TODO-- automatic tune proposal width
    # such that the acceptance rate is between
    # 0.2 and 0.8.

    infer_std = False

    if infer_std:
        proposal_width = 0.1
    else:
        proposal_width = 0.2

    if kwargs["branch"] == "candidate_selection":
        prior_type = "jeffrey"
        prior_width = None
        prior_mean = None

    elif kwargs["branch"] == "safety_test":
        # use candidate zhat mean to construct prior for safety test
        if kwargs["use_candidate_prior"]:
            prior_type = "normal"
            prior_mean = kwargs["zhat_mean"]
            # print(prior_mean)
            prior_width = 0.5
        else:
            prior_type = "jeffrey"
            prior_width = None
            prior_mean = None


    likelihood_ratio = get_likelihood_ratio(statistic_name, zhat, datasize, infer_std)

    mh = MetropolisHastings(proposal_width, prior_type, prior_mean, prior_width, likelihood_ratio, infer_std)
    samples, _ = mh.run(N=100000, skip_interval=10, burn_in=3000)

    

    # if kwargs["branch"] == "safety_test":
    # import matplotlib.pyplot as plt
    # from scipy.stats import t
    # plt.hist(samples, bins=100, density=True, label="posterior", alpha=0.5)
    # plt.axvline(np.quantile(samples, 0.9, method="inverted_cdf"))

    # print(np.quantile(samples, 0.95), np.mean(zhat) + np.std(zhat) / np.sqrt(datasize) * t.ppf(0.95, datasize - 1))

    # normal_samples = np.random.normal(loc=np.mean(zhat), scale=np.std(zhat) / np.sqrt(datasize), size=(10000,))
    # plt.hist(normal_samples, bins=100, density=True, label="gaussian", alpha=0.5)

    # plt.hist(zhat, bins=100, density=True, label="zhat", alpha=0.5)
    # plt.legend()
    # plt.show()


    # print(np.quantile(samples, 0.9, method="inverted_cdf") )


    return samples
