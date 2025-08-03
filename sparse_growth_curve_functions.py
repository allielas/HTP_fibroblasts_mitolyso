# Adapted from:**Authors**: Chuankai Cheng and J. Cameron Thrash(*)
# Department of Biological Sciences, University of Southern California, Los Angeles, CA, USA
# package for numerical analysis: numpy
import numpy as np

# package for plotting: plt
from matplotlib import pyplot as plt
import pandas as pd

# statistical analysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy import stats

# methods for 1-D interpolation: interp1d
from scipy.interpolate import interp1d


def myLinearRegression_CB(x, y, x_fit, one_order=10):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    print("\nFitting data:")
    print("x = ", x)
    print("y = ", y)

    corr = np.corrcoef(x, y)[0][1]
    print("|corr|=", np.abs(corr))

    if (
        ((np.abs(corr) < 0.80) or (len(y) < 4))
        and ((np.abs(corr) < 0.90) or (len(y) < 3))
        and ((np.max(y) - np.min(y)) < np.log2(one_order))
    ):
        comp_y = np.median(y) * np.ones(len(y))
        pre_y = np.median(y) * np.ones(len(x_fit))
        doubling_rate = 1e-6

    else:
        # Robust linear model estimation using RANSAC
        X = x.reshape(-1, 1)
        if len(y) > 4:
            try:
                reg = RANSACRegressor()
                reg.fit(X, y)
                doubling_rate = reg.estimator_.coef_[0]
                inlier_mask = reg.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)

            except ValueError:
                print(
                    "RANSAC could not find a valid consensus set.\n",
                    "Running regular linear regression.",
                )
                reg = LinearRegression()
                reg.fit(X, y)
                doubling_rate = reg.coef_[0]
        else:
            reg = LinearRegression()
            reg.fit(X, y)
            doubling_rate = reg.coef_[0]

        pre_y = reg.predict(x_fit.reshape(-1, 1))
        comp_y = reg.predict(X)

    sigma = np.sqrt(np.sum((comp_y - y) ** 2 / (len(x) - 1)))

    T_95 = stats.t.ppf(0.95, len(x) - 1)

    G = np.sqrt(1 / len(x) + (x_fit - np.mean(x)) ** 2 / sum((x - np.mean(x)) ** 2))

    ci = sigma * T_95 * G

    # sigma**2/np.sum()

    return (doubling_rate, pre_y, ci)


def preprocessing_growth_curve(time, cell_density):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    t = np.array(time)
    X = np.array(cell_density)

    # for i in range(len(time)-3):
    #  t=np.r_[t,np.median(time[[i,i+1,i+2]])]
    #  X=np.r_[X, np.median(cell_density[[i,i+1, i+2]])]

    X = X[np.argsort(t)]
    t = t[np.argsort(t)]

    # You might get multiple cell counts for the same sample at one time.
    # Here, I merge the cell densities by their mean:
    # using the function "np.unique"
    # I get rid of the duplicated time points
    t1 = np.unique(t)
    X1 = []
    for tt in t1:
        temporary_cell_densities = X[t == tt]
        # Getting the mean cell density at a time point
        X1.append(np.median(temporary_cell_densities))
    X1 = np.array(X1)

    return t1, X1


def phase_seperations(t, X, max_depth=1):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    gamma = np.diff(np.log2(X)) / np.diff(t)
    gamma = np.r_[gamma, gamma, gamma]
    # gamma=np.r_[gamma[0], gamma]
    t_gamma = np.r_[
        t[0:-1], np.array([(t[i] + t[i + 1]) / 2 for i in range(len(t) - 1)]), t[1::]
    ]

    gamma_2 = np.diff(np.log2(X)[::2]) / np.diff(t[::2])
    gamma_2 = np.r_[gamma_2, gamma_2, gamma_2]
    t_gamma_2 = np.r_[
        t[::2][0:-1],
        [(t[::2][i] + t[::2][i + 1]) / 2 for i in range(len(t[::2]) - 1)],
        t[::2][1::],
    ]

    gamma_2_1 = np.diff(np.log2(X)[1::2]) / np.diff(t[1::2])
    gamma_2_1 = np.r_[gamma_2_1, gamma_2_1, gamma_2_1]
    t_gamma_2_1 = np.r_[
        t[1::2][0:-1],
        [(t[1::2][i] + t[1::2][i + 1]) / 2 for i in range(len(t[1::2]) - 1)],
        t[1::2][1::],
    ]

    # gamma_3=np.diff(np.log2(X)[::3])/np.diff(t[::3])
    # gamma_3=np.r_[gamma_3, gamma_3, gamma_3]
    # t_gamma_3=np.r_[t[::3][0:-1],
    #                [(t[::3][i]+t[::3][i+1])/2 for i in range(len(t[::3])-1)],
    #                t[::3][1::]]

    all_t_gamma = np.r_[t_gamma, t_gamma_2, t_gamma_2_1]
    all_gamma = np.r_[gamma, gamma_2, gamma_2_1]

    all_gamma = all_gamma[np.argsort(all_t_gamma)]
    all_t_gamma = np.sort(all_t_gamma)

    all_gamma = np.array(
        [
            np.median(all_gamma[[i, i + 1, i + 2, i + 3]])
            for i in range(len(all_t_gamma) - 4)
        ]
    )
    all_t_gamma = np.array(
        [
            np.median(all_t_gamma[[i, i + 1, i + 2, i + 3]])
            for i in range(len(all_t_gamma) - 4)
        ]
    )

    sel_t_gamma = np.unique(all_t_gamma)
    sel_gamma = []
    for stg in sel_t_gamma:
        sel_gamma.append(np.mean(all_gamma[all_t_gamma == stg]))
    sel_gamma = np.array(sel_gamma)

    # print(sel_gamma)
    # print(sel_t_gamma)
    # By default, max_depth = 1
    # Because for a standard growth curve (no diauxic shift) without death phase,
    # there would only be two states:
    # 1. Not growing (lag phase and stationary), growth rate is close to 0;
    # 2. Growing exponentially at an almost constant rate.

    regr_1 = DecisionTreeRegressor(max_depth=max_depth)
    regr_1.fit(sel_t_gamma.reshape(-1, 1), sel_gamma)

    t_fit = np.arange(0.0, t[-1], 0.01)[:, np.newaxis]
    gamma_fit = regr_1.predict(t_fit)

    # print(gamma_fit)
    # We find the state transition point
    gamma_fit_diff = np.diff(gamma_fit)
    inflection_points = t_fit[1::][gamma_fit_diff != 0]

    all_starting_time = np.r_[[0], inflection_points.reshape(1, -1)[0], t_fit[-1]]

    return all_starting_time


def phases_exponential_fit(phases_points, t, X, one_order):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    all_starting_time = phases_points

    all_doubling_rates = []
    all_fit_time = []
    all_fit_cell_density = []
    all_fit_conf_band = []

    print("All phases points", all_starting_time)
    t_1 = np.arange(0.0, t[-1], 0.01)[:, np.newaxis]

    for i in range(len(all_starting_time) - 1):
        start_t = all_starting_time[i]
        end_t = all_starting_time[i + 1]

        # print('Time period: ', start_t, 'hours  ---', end_t, 'hours')
        # Chooseing the time period:
        sel_bool = (t >= start_t - 1) & (t <= end_t + 1)
        # if np.sum(sel_bool)<3:
        #  if np.where(sel_bool)[0][0]!=0 and np.where(sel_bool)[0][-1]!=len(sel_bool)-1:
        #    sel_bool[np.where(sel_bool)[0][0]-1]=True
        #    sel_bool[np.where(sel_bool)[0][-1]+1]=True
        #  elif np.where(sel_bool)[0][0]!=0:
        #    sel_bool[np.where(sel_bool)[0][0]-1]=True
        #  elif np.where(sel_bool)[0][-1]!=len(sel_bool)-1:
        #    sel_bool[np.where(sel_bool)[0][-1]+1]=True

        if np.sum(sel_bool) >= 2:
            sel_t = t[sel_bool]
            sel_X = X[sel_bool]

            # print(sel_t, sel_X)

            fit_bool = (t_1 >= start_t - 1) & (t_1 <= end_t + 1)
            sel_t_1 = t_1[fit_bool]

            (dr, pre_X_1, ci) = myLinearRegression_CB(sel_t, np.log2(sel_X), sel_t_1)

            all_doubling_rates.append(dr)
            all_fit_time.append(sel_t_1)
            all_fit_cell_density.append(2**pre_X_1)
            all_fit_conf_band.append(2**ci)

            print("Doubling rate:", dr, "doubling/hour")
            print("\n")

        else:
            print("No data point in this time period, not fitting.")

    return (all_doubling_rates, all_fit_time, all_fit_cell_density, all_fit_conf_band)


def growth_death_rate_decision(all_fit_cell_density, all_fit_time, all_doubling_rates):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    # all_starting_time=phases_points
    all_acrossing_orders = []
    for i in range(len(all_fit_cell_density)):
        start_t = all_fit_time[i][0]
        end_t = all_fit_time[i][-1]

        print("\nTime period: ", start_t, "hours  ---", end_t, "hours")

        afc = all_fit_cell_density[i]
        acrossing_orders = np.log10(afc[-1]) - np.log10(afc[0])
        all_acrossing_orders.append(acrossing_orders)

        print("Doubling rate:", all_doubling_rates[i], "doubling/hour")
        print("Number of orders acrossing:", acrossing_orders)

    selected_doubling_rate = 0
    selected_fit_time = 0
    selected_fit_cell_density = all_fit_cell_density[0][0]

    selected_doubling_rate_d = 0
    selected_fit_time_d = all_fit_time[-1][-1]
    selected_fit_cell_density_d = all_fit_cell_density[-1][-1]

    # Growth phase:
    if max(all_acrossing_orders) > 0:
        selected_i = np.argmax(all_acrossing_orders)
        selected_doubling_rate = all_doubling_rates[selected_i]
        selected_fit_time = all_fit_time[selected_i]
        selected_fit_cell_density = all_fit_cell_density[selected_i]

    # Death phase:
    if min(all_acrossing_orders) < 0:
        selected_i_d = np.argmin(all_acrossing_orders)
        selected_doubling_rate_d = all_doubling_rates[selected_i_d]
        selected_fit_time_d = all_fit_time[selected_i_d]
        selected_fit_cell_density_d = all_fit_cell_density[selected_i_d]

    return (
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    )


def fit_growth_curve(time, cell_density, one_order=10, decision_tree_depth=1):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    t1, X1 = preprocessing_growth_curve(time, cell_density)
    phases_points = phase_seperations(t1, X1, max_depth=decision_tree_depth)

    print(phases_points)

    (all_doubling_rates, all_fit_time, all_fit_cell_density, all_fit_conf_band) = (
        phases_exponential_fit(phases_points, t1, X1, one_order)
    )

    (
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    ) = growth_death_rate_decision(
        all_fit_cell_density, all_fit_time, all_doubling_rates
    )

    return (
        all_fit_time,
        all_fit_cell_density,
        all_fit_conf_band,
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    )


def fit_growth_curve_ransac_method(time, cell_density):
    """
    :Authors:
      Chuankai Cheng <chuankai@usc.edu> and J. Cameron Thrash <thrash@usc.edu>
    :License:
      MIT
    :Version:
      1.0
    :Date:
      2021-03-17
    :Repository: https://github.com/thrash-lab/sparse-growth-curve
    """
    outlier_mask = np.array([True for i in range(len(cell_density))])
    t_ransac = time[outlier_mask]
    cd_ransac = cell_density[outlier_mask]

    all_ransac_periods = []
    all_ransac_periods_cd = []
    all_ransac_periods_order = []
    all_ransac_periods_doubling_rate = []
    p = 1
    while len(t_ransac) > 2:
        reg = RANSACRegressor()
        reg.fit(t_ransac.reshape(-1, 1), np.log2(cd_ransac))
        inlier_mask = reg.inlier_mask_

        print("\nPeriod", p, ":")
        cd_fit = 2 ** (reg.predict(t_ransac[inlier_mask].reshape(-1, 1)))

        all_ransac_periods.append(t_ransac[inlier_mask])
        all_ransac_periods_cd.append(cd_fit)

        print(
            "Time=",
            t_ransac[inlier_mask],
            "\nCell density=",
            cd_ransac[inlier_mask],
            "\nFit cell density=",
            cd_fit,
        )
        print("Doubling rate=", reg.estimator_.coef_[0], "Doubling/hour")
        orders = np.log10(cd_fit[-1] / cd_fit[0])
        print("Orders acorssing=", orders)
        all_ransac_periods_doubling_rate.append(reg.estimator_.coef_[0])
        all_ransac_periods_order.append(orders)

        outlier_mask = np.logical_not(inlier_mask)

        t_ransac = t_ransac[outlier_mask]
        cd_ransac = cd_ransac[outlier_mask]

        p += 1

    ransac_selected_doubling_rate = all_ransac_periods_doubling_rate[
        np.argmax(all_ransac_periods_order)
    ]
    ransac_selected_doubling_rate_d = all_ransac_periods_doubling_rate[
        np.argmin(all_ransac_periods_order)
    ]

    ransac_selected_periods = all_ransac_periods[np.argmax(all_ransac_periods_order)]
    ransac_selected_periods_d = all_ransac_periods[np.argmin(all_ransac_periods_order)]

    ransac_selected_periods_cd = all_ransac_periods_cd[
        np.argmax(all_ransac_periods_order)
    ]
    ransac_selected_periods_cd_d = all_ransac_periods_cd[
        np.argmin(all_ransac_periods_order)
    ]

    return (
        all_ransac_periods,
        all_ransac_periods_cd,
        all_ransac_periods_order,
        all_ransac_periods_doubling_rate,
        ransac_selected_doubling_rate,
        ransac_selected_doubling_rate_d,
        ransac_selected_periods,
        ransac_selected_periods_d,
        ransac_selected_periods_cd,
        ransac_selected_periods_cd_d,
    )


def plot_confluence_curve_estimations(
    df,
    time_col,
    cell_density_col,
    one_order=10,
    tree_depth=1,
    curve_name="",
    savepath="",
):
    """_summary_

    Args:
        df (DataFrame): the pandas dataframe containing growth curve data
        time (str): the column name with  elapsed times
        cell_density (str): the col name with cell density in any units
    """
    time = df[time_col].to_numpy()
    cell_density = df[cell_density_col].to_numpy()

    (
        all_fit_time,
        all_fit_cell_density,
        all_fit_conf_band,
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    ) = fit_growth_curve(
        time, cell_density, one_order=one_order, decision_tree_depth=tree_depth
    )

    (
        all_ransac_periods,
        all_ransac_periods_cd,
        all_ransac_periods_order,
        all_ransac_periods_doubling_rate,
        ransac_selected_doubling_rate,
        ransac_selected_doubling_rate_d,
        ransac_selected_periods,
        ransac_selected_periods_d,
        ransac_selected_periods_cd,
        ransac_selected_periods_cd_d,
    ) = fit_growth_curve_ransac_method(time, cell_density)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    for i in range(len(all_fit_time)):
        # plt.plot(all_fit_time[i], all_fit_cell_density[i], 'k--')
        plt.fill_between(
            all_fit_time[i],
            all_fit_cell_density[i] * (all_fit_conf_band[i]),
            all_fit_cell_density[i] / (all_fit_conf_band[i]),
            color="k",
            alpha=0.1,
        )

    plt.plot(selected_fit_time, selected_fit_cell_density, "k-", linewidth=2)

    try:
        selected_doubling_time = selected_doubling_rate**-1
        plt.text(
            np.median(selected_fit_time),
            2 ** np.median(np.log2(selected_fit_cell_density)),
            str(np.round(selected_doubling_time, 2)) + "hours/doubling",
        )
    except ZeroDivisionError as e:
        print(f"{e}, Growth rate is zero, using a placeholder")
        plt.text(
            np.median(selected_fit_time),
            2 ** np.median(np.log2(selected_fit_cell_density)),
            str(np.round(selected_doubling_rate, 2)) + "doublings/hr",
        )

    plt.plot(selected_fit_time_d, selected_fit_cell_density_d, "k--", linewidth=2)

    plt.text(
        np.median(selected_fit_time_d),
        2 ** np.median(np.log2(selected_fit_cell_density_d)),
        str(np.round(selected_doubling_rate_d, 2)) + "doubling/hr",
    )

    plt.plot(time, cell_density, "ro", mfc="none")

    plt.xlabel("Time (h)")
    plt.ylabel("Cell density (confluence)")

    plt.yscale("log")
    plt.title("Decision tree method")

    plt.subplot(122)
    for i in range(len(all_ransac_periods)):
        plt.plot(all_ransac_periods[i], all_ransac_periods_cd[i], "r-")
        try:
            plt.text(
                np.median(all_ransac_periods[i]),
                2 ** np.median(np.log2(all_ransac_periods_cd[i])),
                str(np.round(1 / all_ransac_periods_doubling_rate[i], 2))
                + "hours/doubling",
            )
        except ZeroDivisionError as e:
            print(f"{e}, Growth rate is zero, using a placeholder")
            plt.text(
                np.median(all_ransac_periods[i]),
                2 ** np.median(np.log2(all_ransac_periods_cd[i])),
                str(np.round(all_ransac_periods_doubling_rate[i], 2))
                + "doublings/hour",
            )

    plt.plot(time, cell_density, "ro", mfc="none")

    plt.xlabel("Time (h)")
    plt.ylabel("Cell density (confluence)")

    plt.yscale("log")
    plt.title("Iterative RANSAC method")

    plt.tight_layout()
    fig_path = f"{savepath}/{curve_name}_methods_comparison.png"
    plt.savefig(fig_path)
    plt.show()


def get_doubling_time_from_decision_trees(
    df, time_col, cell_density_col, one_order=10, tree_depth=1
):
    """_summary_

    Args:
        df (DataFrame): the pandas dataframe containing growth curve data
        time (str): the column name with  elapsed times
        cell_density (str): the col name with cell density in any units

    Return:
        selected_doubling_time (float): The doubling time in hours from the curve fit estimation
    """
    time = df[time_col].to_numpy()
    cell_density = df[cell_density_col].to_numpy()

    (
        all_fit_time,
        all_fit_cell_density,
        all_fit_conf_band,
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    ) = fit_growth_curve(
        time, cell_density, one_order=one_order, decision_tree_depth=tree_depth
    )
    try:
        selected_doubling_time = selected_doubling_rate**-1
    except ZeroDivisionError as e:
        print(f"{e}, Growth rate is zero, returning a placeholder")
        return 200.0
    return selected_doubling_time


def get_doubling_time_from_ransac_method(df, time_col, cell_density_col):
    """_summary_

    Args:
        df (DataFrame): the pandas dataframe containing growth curve data
        time (str): the column name with  elapsed times
        cell_density (str): the col name with cell density in any units

    Return:
        selected_doubling_time (float): The doubling time in hours from the curve fit estimation using the RANSAC method
    """
    time = df[time_col].to_numpy()
    cell_density = df[cell_density_col].to_numpy()
    (
        all_ransac_periods,
        all_ransac_periods_cd,
        all_ransac_periods_order,
        all_ransac_periods_doubling_rate,
        ransac_selected_doubling_rate,
        ransac_selected_doubling_rate_d,
        ransac_selected_periods,
        ransac_selected_periods_d,
        ransac_selected_periods_cd,
        ransac_selected_periods_cd_d,
    ) = fit_growth_curve_ransac_method(time, cell_density)
    try:
        selected_doubling_time = ransac_selected_doubling_rate**-1
    except ZeroDivisionError as e:
        print(f"{e}, Growth rate is zero, returning a placeholder")
        return 200.0
    return selected_doubling_time


def get_doubling_rates_times_both_methods(
    df,
    time_col,
    cell_density_col,
    one_order=10,
    tree_depth=1,
    curve_name="",
    savepath="",
):
    """_summary_

    Args:
        df (DataFrame): the pandas dataframe containing growth curve data
        time (str): the column name with  elapsed times
        cell_density (str): the col name with cell density in any units
    """
    time = df[time_col].to_numpy()
    cell_density = df[cell_density_col].to_numpy()

    (
        all_fit_time,
        all_fit_cell_density,
        all_fit_conf_band,
        selected_doubling_rate,
        selected_fit_time,
        selected_fit_cell_density,
        selected_doubling_rate_d,
        selected_fit_time_d,
        selected_fit_cell_density_d,
    ) = fit_growth_curve(
        time, cell_density, one_order=one_order, decision_tree_depth=tree_depth
    )

    try:
        selected_doubling_time = selected_doubling_rate**-1
    except ZeroDivisionError as e:
        print(f"{e}, Growth rate is zero, using a placeholder")
        selected_doubling_time = 200.0

    (
        all_ransac_periods,
        all_ransac_periods_cd,
        all_ransac_periods_order,
        all_ransac_periods_doubling_rate,
        ransac_selected_doubling_rate,
        ransac_selected_doubling_rate_d,
        ransac_selected_periods,
        ransac_selected_periods_d,
        ransac_selected_periods_cd,
        ransac_selected_periods_cd_d,
    ) = fit_growth_curve_ransac_method(time, cell_density)
    try:
        ransac_selected_doubling_time = ransac_selected_doubling_rate**-1
    except ZeroDivisionError as e:
        print(f"{e}, Growth rate is zero, returning a placeholder")
        ransac_selected_doubling_time = 200.0

    return {
        "Doubling_Time_Trees": selected_doubling_time,
        "Exponential_Doubling_Rate_Trees": selected_doubling_rate,
        "Selected_Times_Trees": selected_fit_time,
        "Selected_Cell_Density_Trees": selected_fit_cell_density,
        "Doubling_Time_RANSAC": ransac_selected_doubling_time,
        "Exponential_Doubling_Rate_RANSAC": ransac_selected_doubling_rate,
        "Selected_Times_RANSAC": ransac_selected_periods,
    }
