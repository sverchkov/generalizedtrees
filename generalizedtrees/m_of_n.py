# Various methods for implementing m-of-n splits
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


from generalizedtrees.constraints import MofN


# TODO: incomplete and untested
def construct_m_of_n_split(
    self,
    best_split,
    best_split_score,
    split_candidates,
    data,
    targets):

    beam = [(best_split_score, best_split)]
    beam_changed = True
    while beam_changed:
        beam_changed = False

        for _, split in beam:
            for new_split in MofN.neighboring_tests(split, split_candidates):
                if self.tests_sig_diff(split, new_split): #+data?
                    new_score = self.split_score(new_split, data, targets)
                    # Pseudocode in paper didn't have this but it needs to be here, right?
                    if len(beam) < self.beam_width:
                        # We're modifying a list while iterating over it, but since we're adding,
                        # this should be ok.
                        beam.append((new_score, new_split))
                        beam_changed = True
                    else:
                        worst = min(beam)
                        if new_score > worst[0]: # Element 0 of the tuple is the score
                            beam[beam.index(worst)] = (new_score, new_split)
                            beam_changed = True
    
    # TODO: literal pruning for test (see pages 57-58)

    return max(beam)[1] # Element 1 of the tuple is the split

# TODO: incomplete and untested
def tests_sig_diff(self, split, new_split):
    raise NotImplementedError