Download folder for the two public files of Kramer et al. (2023), THINGS_Memorability_Scores.csv and
Concept_to_category_linking.csv (osf.io/5a7z6). From the repository root,

    python external/download_external.py

puts them here, checks their sizes and md5 sums, and checks the tables in shared/things/ against them.
No module needs these files: the repository includes the scores and categories that the analyses use.
