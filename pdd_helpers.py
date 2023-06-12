import amd
import collections
from scipy.spatial.distance import squareform, pdist
import numpy as np


def _collapse_into_groups(overlapping):
    """The vector `overlapping` indicates for each pair of items in a set whether
    or not the items overlap, in the shape of a condensed distance matrix. Returns
    a list of groups of indices where all items in the same group overlap."""

    overlapping = squareform(overlapping)
    group_nums = {}  # row_ind: group number
    group = 0
    for i, row in enumerate(overlapping):
        if i not in group_nums:
            group_nums[i] = group
            group += 1

            for j in np.argwhere(row).T[0]:
                if j not in group_nums:
                    group_nums[j] = group_nums[i]

    groups = collections.defaultdict(list)
    for row_ind, group_num in sorted(group_nums.items()):
        groups[group_num].append(row_ind)
    groups = list(groups.values())

    return groups


def custom_PDD(
        periodic_set,
        k: int,
        lexsort: bool = False,
        collapse: bool = False,
        collapse_tol: float = 1e-4,
        return_row_groups: bool = True,
        constrained: bool = True,
) -> np.ndarray:
    """The PDD of a periodic set (crystal) up to k.
    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet`  tuple of :class:`numpy.ndarray` s
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form and a square unit cell.
    k : int
        The returned PDD has k+1 columns, an additional first column for row weights.
        k is the number of neighbours considered for each atom in the unit cell
        to make the PDD.
    lexsort : bool, default True
        Lexicographically order the rows. Default True.
    collapse: bool, default True
        Collapse repeated rows (within the tolerance ``collapse_tol``). Default True.
    collapse_tol: float, default 1e-4
        If two rows have all elements closer than ``collapse_tol``, they are merged and
        weights are given to rows in proportion to the number of times they appeared.
        Default is 0.0001.
    return_row_groups: bool, default False
        Return data about which PDD rows correspond to which points.
        If True, a tuple is returned ``(pdd, groups)`` where ``groups[i]``
        contains the indices of the point(s) corresponding to ``pdd[i]``.
        Note that these indices are for the asymmetric unit of the set, whose
        indices in ``periodic_set.motif`` are accessible through
        ``periodic_set.asymmetric_unit``.
    Returns
    -------
    numpy.ndarray
        A :class:`numpy.ndarray` with k+1 columns, the PDD of ``periodic_set`` up to k.
        The first column contains the weights of rows. If ``return_row_groups`` is True,
        returns a tuple (:class:`numpy.ndarray`, list).
    Examples
    --------
    Make list of PDDs with ``k=100`` for crystals in mycif.cif::
        pdds = []
        for periodic_set in amd.CifReader('mycif.cif'):
            # do not lexicographically order rows
            pdds.append(amd.PDD(periodic_set, 100, lexsort=False))
    Make list of PDDs with ``k=10`` for crystals in these CSD refcode families::
        pdds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            # do not collapse rows
            pdds.append(amd.PDD(periodic_set, 10, collapse=False))
    Manually pass a periodic set as a tuple (motif, cell)::
        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        cubic_amd = amd.PDD((motif, cell), 100)
    """

    motif, cell, asymmetric_unit, weights = _extract_motif_cell(periodic_set)
    dists, cloud, inds = amd.nearest_neighbours(motif, cell, asymmetric_unit, k)
    groups = [[i] for i in range(len(dists))]

    if collapse and collapse_tol >= 0:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping <= collapse_tol
        types_match = pdist(periodic_set.types.reshape((-1, 1))) == 0
        neighbors_match = (pdist(periodic_set.types[inds % periodic_set.types.shape[0]]) == 0)

        if constrained:
            overlapping = overlapping & types_match & neighbors_match
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            dists = np.array([np.average(dists[group], axis=0) for group in groups])

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        if return_row_groups:
            groups = [groups[i] for i in lex_ordering]
        pdd = pdd[lex_ordering]

    if return_row_groups:
        return pdd, groups, inds, cloud
    else:
        return pdd, inds, cloud


def _extract_motif_cell(pset: amd.PeriodicSet):
    """``pset`` is either a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or a tuple of
    :class:`numpy.ndarray` s (motif, cell). If possible, extracts the
    asymmetric unit and wyckoff multiplicities.
    """

    if isinstance(pset, amd.PeriodicSet):
        motif, cell = pset.motif, pset.cell
        asym_unit = pset.asymmetric_unit
        wyc_muls = pset.wyckoff_multiplicities
        if asym_unit is None or wyc_muls is None:
            asymmetric_unit = motif
            weights = np.full((len(motif),), 1 / len(motif))
        else:
            asymmetric_unit = pset.motif[asym_unit]
            weights = wyc_muls / np.sum(wyc_muls)
    else:
        motif, cell = pset
        asymmetric_unit = motif
        weights = np.full((len(motif),), 1 / len(motif))

    return motif, cell, asymmetric_unit, weights
