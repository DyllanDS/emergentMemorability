"""
neighbor_share.py: how often a concept's nearest neighbors in the SPoSE space belong to its own
higher-level category (SI Appendix, Section 5: "62% of a concept's ten nearest SPoSE neighbors belong
to its own category, against 4% by chance"). Runs on public data included in this repository; no
trial file needed.

WHAT IT DOES
    1. Similarity of two concepts = the Pearson correlation of their 49 SPoSE feature values (the
       measure every Section 5 script uses).
    2. For each concept, its ten most similar other concepts are its ten nearest neighbors.
    3. Over the concepts that belong to one of the named higher-level categories (category code not 0;
       1,324 concepts), the share of their ten nearest neighbors that carry the same category code.
       A neighbor without a named category never counts as a match.
    4. Chance: a neighbor drawn from one of the 27 higher-level THINGS categories (Hebart et al. 2019)
       at random belongs to a given one with probability 1/27.
    As in the other Section 5 scripts, concepts are numbered as in the Kramer et al. files
    (alphabetical order of the concept names): row r of the SPoSE file and row r of the category table
    in that numbering are taken as the same concept.

INPUT   shared/things/spose_embedding_49d_sorted.txt   (Hebart et al. 2020, 1,854 x 49)
        shared/things/concept_categories.csv           (the public category file, via
                                                        a5_common.concept_categories_alphabetical())
OUTPUT  outputs/neighbor_share.txt
SEEDS   none.  RUNTIME  seconds.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/neighbor_share.py
"""
import numpy as np

import a5_common

NUMBER_OF_NEIGHBORS = 10
NUMBER_OF_THINGS_CATEGORIES = 27     # the higher-level categories of THINGS (Hebart et al. 2019)


def compute():
    """Returns a dictionary with the share, the chance level and the counts behind them."""
    features = np.loadtxt(a5_common.spose_file())
    number_of_concepts = features.shape[0]

    # similarity of every pair of concepts; a concept is not its own neighbor
    similarity = np.corrcoef(features)
    np.fill_diagonal(similarity, -np.inf)

    category_table = a5_common.concept_categories_alphabetical()
    category_code = category_table['category'].to_numpy()
    if len(category_code) != number_of_concepts:
        raise SystemExit('the category table and the SPoSE file have different numbers of concepts')

    concepts_with_a_named_category = 0
    same_category_neighbors = 0
    for concept in range(number_of_concepts):
        own_category = category_code[concept]
        if own_category == 0:
            continue                                  # no named higher-level category
        concepts_with_a_named_category = concepts_with_a_named_category + 1
        ranked = np.argsort(-similarity[concept])     # most similar first
        nearest = ranked[:NUMBER_OF_NEIGHBORS]
        for neighbor in nearest:
            if category_code[neighbor] == own_category:
                same_category_neighbors = same_category_neighbors + 1

    share = same_category_neighbors / (concepts_with_a_named_category * NUMBER_OF_NEIGHBORS)
    chance = 1.0 / NUMBER_OF_THINGS_CATEGORIES
    return dict(share=share, chance=chance, concepts=number_of_concepts,
                concepts_with_a_named_category=concepts_with_a_named_category,
                same_category_neighbors=same_category_neighbors)


def main():
    result = compute()
    lines = [
        "NEAREST SPoSE NEIGHBORS IN THE CONCEPT'S OWN HIGHER-LEVEL CATEGORY",
        'concepts: %d; with a named higher-level category: %d'
        % (result['concepts'], result['concepts_with_a_named_category']),
        'ten nearest neighbors (Pearson correlation of the SPoSE rows), same category: %d of %d = %.4f (%.1f%%)'
        % (result['same_category_neighbors'], result['concepts_with_a_named_category'] * NUMBER_OF_NEIGHBORS,
           result['share'], 100 * result['share']),
        'chance (1 of the %d higher-level THINGS categories): %.4f (%.1f%%)'
        % (NUMBER_OF_THINGS_CATEGORIES, result['chance'], 100 * result['chance']),
    ]
    path = a5_common.OUTPUTS / 'neighbor_share.txt'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('wrote', a5_common.relative(path))


if __name__ == '__main__':
    main()
