#!/usr/bin/python
"""bins first level results dataframe to reduce the data size
"""
import pandas as pd
import numpy as np
import argparse


def _bin_eccen(eccens):
    eccen_min = int(np.floor(eccens.min()))
    eccen_max = int(np.ceil(eccens.max()))
    return pd.cut(eccens, range(eccen_min, eccen_max+1),
                  labels=["%02d-%02d" % (i, i+1) for i in range(eccen_min, eccen_max)])


def _bin_angle_quarters(angles):
    new_angles = pd.cut(angles, [0, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4, 2*np.pi],
                        labels=['right upper', 'upper', 'left', 'lower', 'right lower'])
    new_angles = new_angles.astype(str)
    new_angles[new_angles == 'right upper'] = 'right'
    new_angles[new_angles == 'right lower'] = 'right'
    return new_angles


def main(df, to_bin=['eccen'], save_path=None, weighted_avg=False):
    """bin first level results dataframe

    when we bin a column, we take the mean of all voxels in the corresponding bins in the other
    columns. for some of these columns, the output won't make sense, so think before making use of
    the output columns. It will make sense for: spatial frequency, amplitude estimates. It won't
    for: hemispheres, angles (if not binning over them). We reset the voxel numbers for the binned
    dataframe, but note that they obviously do not correspond to the same thing as before binning.

    to_bin: list that contains some of: 'eccen', 'angle'. Which columns to bin by. Currently, we
    bin eccentricty in one degree bins and angle in quarters of the visual field. Note that in
    either of these cases, we collapse across cortical hemispheres as appropriate (e.g., if we bin
    into quarters of the visual field, the left and right visual field will not be collapsed across
    hemispheres, but the upper and lower visual fields will). Note that this must contain
    *something*.

    save_path: str, optional. Where to save the binned dataframe.

    weighted: boolean, optional. If True, we weight all averages by the precision column (which is
    the inverse of the variance of the amplitude estimates for each voxel).
    """
    df = df.copy()
    if len(to_bin) == 0:
        raise Exception("You must bin by something!")
    if 'eccen' in to_bin:
        df['eccen'] = _bin_eccen(df.eccen.values)
    if 'angle' in to_bin:
        df['angle'] = _bin_angle_quarters(df.angle.values)
    if 'bootstrap_num' in df.columns:
        to_bin.append('bootstrap_num')
    # these columns will have unique combinations of values that correspond to "binned voxels"
    voxel_id_cols = ['varea'] + to_bin
    to_bin.extend(['stimulus_class', 'varea'])
    # We loop through all columns in the groupby, checking if they're numeric.  if they are, we
    # average all those values (setting weights to either None or reliability, depending on whether
    # user selected to weight them or not); if it's not, check whether there's one unique value. if
    # there is, use that. if not, return None.
    data_dict = {}
    gb = df.groupby(to_bin)
    for col in df.columns:
        if col not in to_bin:
            if np.issubdtype(df[col].dtype, np.number):
                if weighted_avg:
                    data_dict[col] = gb.apply(lambda x: np.average(x[col], weights=x.precision))
                else:
                    data_dict[col] = gb.apply(lambda x: np.average(x[col], weights=None))
            elif (gb[col].nunique() == 1).all():
                data_dict[col] = gb.apply(lambda x: x.stimulus_superclass.unique()[0])
            else:
                data_dict[col] = None
    df = pd.concat(data_dict, 1).reset_index()
    df = df.set_index(voxel_id_cols)
    df_index = df.index.unique()
    voxel_ids = pd.DataFrame(data={'voxel': range(len(df_index))}, index=df_index)
    df['voxel'] = voxel_ids['voxel']
    df = df.reset_index()
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Load in the first level results dataframe and bin by something (currently,"
                     " some combination of angle and eccentricity). Note that you must bin by"
                     " *something*, so at least one of the following must be passed: --angle, "
                     "--eccen"),
        formatter_class=CustomFormatter)
    parser.add_argument("first_level_results_path",
                        help=("Path to the unbinned first level results dataframe"))
    parser.add_argument("save_path",
                        help="Path to save the resulting binned results dataframe at.")
    parser.add_argument("--eccen", action="store_true",
                        help=("Whether to bin the eccentricites into integer increments."))
    parser.add_argument("--angle", action="store_true",
                        help=("Whether to bin the angles into quarters of the visual field (upper,"
                              " lower, right, left)."))
    parser.add_argument("--weighted_avg", action="store_true",
                        help=("Whether to weight the averages (of numeric columns) by each voxels'"
                              " reliability"))
    args = vars(parser.parse_args())
    to_bin = []
    if args.pop('eccen'):
        to_bin.append('eccen')
    if args.pop('angle'):
        to_bin.append('angle')
    df = pd.read_csv(args.pop('first_level_results_path'))
    main(df, to_bin=to_bin, **args)
