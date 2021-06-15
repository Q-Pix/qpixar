# -----------------------------------------------------------------------------
#  qpixtools.py
#
#  Example for reading from ROOT file
#   * Author: Everybody is an author!
#   * Creation date: 15 June 2021
# -----------------------------------------------------------------------------

def fill_particle_map(track_ids):

    """
    Maps MC particle track IDs to MC particle indices for an event.

    Returns a dictionary where the MC particle track ID is the key
    and the MC particle index is the value.

    Parameters 
    ---------- 
    track_ids : array_like
        1-D array of track IDs.

    Returns
    -------
    out : dictionary
        Map of track IDs to indices for an event.

    """

    return { track_ids[idx] : idx for idx in range(len(track_ids)) }

def fill_shower_particle_map(track_id, daughter_track_ids, particle_map,
                             shower_particle_map):

    """
    Returns a dictionary where the MC particle track ID is the key
    and the MC particle index is the value.

    Parameters
    ----------
    track_id : int
        Track ID of the MC particle (root node)

    daughter_track_ids : array-like
        Jagged array of daughter track IDs where the rows correspond
        to the parent MC particles (same index used as the MC
        particle track ID).

    particle_map : dict
        Map of MC particle track IDs to indices for an event.  This
        should already be filled.

    shower_particle_maps : dict
        Map of MC particle track IDs to indices for an event.  This
        is to be filled and/or modified by this function.

    Returns
    -------
    out : None

    Notes
    -----
    This is a depth-first traversal.

               0 
              / \
             1   6
            / \   \
           2   5   7
          / \     / \
         3   4   8   9

    """

    # get index of MC particle from the MC particle map
    particle_idx = particle_map[track_id]

    # add index of MC particle to the MC shower particle map
    shower_particle_map[track_id] = particle_idx

    # loop over track IDs of daughter particles
    for daughter_track_id in daughter_track_ids[particle_idx]:

        # recur
        fill_shower_particle_map(daughter_track_id, daughter_track_ids,
                                 particle_map, shower_particle_map)

    return

