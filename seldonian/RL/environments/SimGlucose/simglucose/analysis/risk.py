import numpy as np
import warnings


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    # BG \in [70, 180] is good.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.arange(1, 500)
    y = []
    for item in x:
        item = np.array([item])
        fBG = 1.509 * (np.log(item) ** 1.084 - 5.381)
        rl = 10 * fBG[fBG < 0] ** 2
        rh = 10 * fBG[fBG > 0] ** 2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        y.append(- (LBGI + HBGI))

    plt.plot(y)
    plt.show()