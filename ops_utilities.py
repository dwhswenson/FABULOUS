try:
    import openpathsampling as paths
except ImportError:
    HAS_OPS = False
else:
    HAS_OPS = True

def _select_trajs(steps):
    """Generator that yields relevant trajectories from OPS steps.
    """
    for step in steps:
        if len(step.active) > 1:
            raise RuntimeError("FABULOUS requires inputs with only one "
                               "sample per step")

        traj = step.active[0].trajectory
        yield traj

def extract_CV(traj, cvs):
    """Create a CV dataframe for FABULOUS from an OPS trajectory and CVs.

    Returns
    -------
    pandas.DataFrame :
        each row is a snapshot
    """
    columns = [cv.name for cv in cvs]
    results = np.array([cv(traj) for cv in cvs])
    return pd.DataFrame(results, columns=columns)


def _get_keep_atoms(topology, keep):
    if isinstance(keep, str):
        return topology.select(keep)
    else:
        return [topology.atom(i) for i in keep]


def extract_MD(trajectory, ref, keep_atoms):
    """Create an MD dataframe for FABULOUS from an OPS trajectory.

    Returns
    -------
    pandas.DataFrame :
        each row is a snapshot, all coordinates in the columns
    """
    traj = trajectory.to_mdtraj().atom_slice(keep_atoms)
    traj.superpose(ref)

    xyz = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    return pd.DataFrame(xyz)


def extract_OPS(steps, ref, keep, cvs):
    """
    """
    keep_atoms = None
    # TODO: add progress bar to this (use OPS progress?)
    for traj in _select_trajs(steps):
        if keep_atoms is None:
            keep_atoms = _get_keep_atoms(keep)

        trj_df = extract_MD(traj, keep_atoms)
        cv_df = extract_CV(traj, cvs)
        yield trj_df, cv_df

