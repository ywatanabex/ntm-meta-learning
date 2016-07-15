import os
import shutil

families_background = \
    ['Japanese_(hiragana)',
     'Futurama',
     'Early_Aramaic',
     'Mkhedruli_(Georgian)',
     'Malay_(Jawi_-_Arabic)',
     'Sanskrit',
     'Asomtavruli_(Georgian)',
     'Armenian',
     'Arcadian',
     'Hebrew',
     'Alphabet_of_the_Magi',
     'Syriac_(Estrangelo)',
     'Blackfoot_(Canadian_Aboriginal_Syllabics)',
     'Balinese',
     'N_Ko',
     'Greek',
     'Burmese_(Myanmar)',
     'Braille',
     'Japanese_(katakana)',
     'Tifinagh',
     'Cyrillic',
     'Tagalog',
     'Gujarati',
     'Latin',
     'Anglo-Saxon_Futhorc',
     'Grantha',
     'Korean',
     'Ojibwe_(Canadian_Aboriginal_Syllabics)',
     'Bengali',
     'Inuktitut_(Canadian_Aboriginal_Syllabics)']

families_evaluation = \
    ['ULOG',
     'Old_Church_Slavonic_(Cyrillic)',
     'Oriya',
     'Mongolian',
     'Syriac_(Serto)',
     'Atlantean',
     'Angelic',
     'Kannada',
     'Tibetan',
     'Ge_ez',
     'Manipuri',
     'Avesta',
     'Atemayar_Qelisayer',
     'Sylheti',
     'Aurek-Besh',
     'Glagolitic',
     'Malayalam',
     'Gurmukhi',
     'Tengwar',
     'Keble']

# split font families
families_train = \
    [('images_background', fam) for (k, fam) in enumerate(families_background) if k % 4 != 0] +\
    [('images_evaluation', fam) for (k, fam) in enumerate(families_evaluation) if k % 4 != 0]
families_test = \
    [('images_background', fam) for (k, fam) in enumerate(families_background) if k % 4 == 0] +\
    [('images_evaluation', fam) for (k, fam) in enumerate(families_evaluation) if k % 4 == 0]

# copy to train dir
if not os.path.exists('train'): os.makedirs('train')
for dn, fam in families_train:
    shutil.copytree(os.path.join(dn, fam), os.path.join('train', fam))

if not os.path.exists('test'): os.makedirs('test')
for dn, fam in families_test:
    shutil.copytree(os.path.join(dn, fam), os.path.join('test', fam))


