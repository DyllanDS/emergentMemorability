"""
spose_quintiles.py: the SPoSE (behavioral space) context effect on the same five-bin scale as the
deep-network analysis (SI Appendix, Section 5, A Similarity Space Without Human Judgments: "0.39 for
SPoSE binned the same way").

WHY
    The deep-network analysis ranks each image's held-out trials into 5 bins (fifths) of crowding and
    reports d' falling by 0.51 from bin 1 to bin 5. The SPoSE analysis of Fig. S13 uses 20 bins: its
    end bins are the most extreme twentieths of each image's trials, so they span a wider range of
    crowding than the end fifths. A like-for-like comparison needs the SPoSE effect on 5 bins.

HOW
    The SPoSE bins are equal-frequency ranks within each image, so every one of the 20 bins holds (to
    within a trial or two per image) the same number of an image's trials. Four adjacent bins therefore
    make one fifth, and the fifth's hit rate and false alarm rate are the plain means of its four bin
    rates (the log-linear corrected rates of sdt_by_context.py). d' = z(hit rate) - z(false alarm rate).
    It uses only the saved 20-bin rates.

INPUT   sdt_ctx_top10_target.npz (keys H and F: the 20 whole-stream bin rates, ctx_top10, target repeats)
        from data/provided/ (the saved run) or outputs/regenerated/ (a run from the trial file)
OUTPUT  outputs/spose_quintiles.txt
SEEDS   none.  RUNTIME  seconds.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/spose_quintiles.py [--source provided]
"""
import argparse

import numpy as np
from scipy.stats import norm

import a5_common

NUMBER_OF_FIFTHS = 5
BINS_PER_FIFTH = 4


def compute(folder):
    """Five-bin hit rates, false alarm rates and d' from the 20-bin file in folder."""
    ventiles = np.load(folder / 'sdt_ctx_top10_target.npz')
    hit_by_ventile = ventiles['H']
    fa_by_ventile = ventiles['F']
    assert len(hit_by_ventile) == 20 and len(fa_by_ventile) == 20

    quintile_hit = []
    quintile_fa = []
    quintile_dprime = []
    for quintile in range(NUMBER_OF_FIFTHS):
        first_ventile = quintile * BINS_PER_FIFTH
        hit_rate = float(np.mean(hit_by_ventile[first_ventile:first_ventile + BINS_PER_FIFTH]))
        fa_rate = float(np.mean(fa_by_ventile[first_ventile:first_ventile + BINS_PER_FIFTH]))
        quintile_hit.append(hit_rate)
        quintile_fa.append(fa_rate)
        quintile_dprime.append(norm.ppf(hit_rate) - norm.ppf(fa_rate))

    # check: the ventile d' recomputed from the saved rates matches the saved ventile d'
    ventile_dprime_recomputed = norm.ppf(hit_by_ventile) - norm.ppf(fa_by_ventile)
    largest_difference = float(np.max(np.abs(ventile_dprime_recomputed - ventiles['dprime'])))
    return dict(hit=quintile_hit, fa=quintile_fa, dprime=quintile_dprime,
                drop=quintile_dprime[0] - quintile_dprime[4],
                twenty_bin_drop=float(ventiles['dprime'][0] - ventiles['dprime'][19]),
                largest_difference=largest_difference)


def main():
    parser = argparse.ArgumentParser(description='SPoSE context effect on five within-image bins')
    parser.add_argument('--source', choices=['provided', 'regenerated'], default=None,
                        help='read the 20-bin file from data/provided/ or outputs/regenerated/ '
                             '(default: regenerated when the trial data are on this machine)')
    args = parser.parse_args()
    source = args.source or a5_common.default_source()
    folder = a5_common.report_folder(source)
    result = compute(folder)

    lines = []
    lines.append('SPoSE WHOLE-STREAM CONTEXT EFFECT ON FIVE WITHIN-IMAGE BINS (from the 20-bin rates)')
    lines.append('source: %s (ctx_top10, target repeats; 20 within-image bins)'
                 % a5_common.relative(folder / 'sdt_ctx_top10_target.npz'))
    lines.append('method: quintile rate = mean of its four ventile rates; d prime = z(H) - z(F)')
    lines.append('check: ventile d prime recomputed from the saved rates vs saved d prime, largest difference %.6f'
                 % result['largest_difference'])
    lines.append('')
    lines.append('bin   hit rate   FA rate    d prime')
    for quintile in range(NUMBER_OF_FIFTHS):
        lines.append('%3d    %.4f    %.4f    %.3f' % (quintile + 1, result['hit'][quintile],
                                                     result['fa'][quintile], result['dprime'][quintile]))
    lines.append('')
    lines.append('five-bin d prime drop, bin 1 minus bin 5: %.3f' % result['drop'])
    lines.append('twenty-bin d prime drop, bin 1 minus bin 20 (for comparison): %.3f' % result['twenty_bin_drop'])
    path = a5_common.OUTPUTS / 'spose_quintiles.txt'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('wrote', a5_common.relative(path))


if __name__ == '__main__':
    main()
