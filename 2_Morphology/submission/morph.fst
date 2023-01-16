ALPHABET = [A-Za-záéíóú<N>] 

%Define intermediate transducers for different morphological classes.
%see https://conjugator.reverso.net/conjugation-spanish.html for reference

% class 1
%Verbs ending in '-er', e.g. 'comer' (to eat) is conjugated in the present tense as
%'como', 'comes', 'come' in singular form (1st-3rd person)
%'comemos', 'coméis', 'comen' in plural (1st-3rd person)

$verb_er$ = <V>:<> (\
  {<1sg>}:{o} |\
  {<2sg>}:{es} |\
  {<3sg>}:{e} |\
  {<1pl>}:{emos} |\
  {<2pl>}:{éis} |\
  {<3pl>}:{en} |\
  {<inf>}:{er}) 


% class 2
%Verbs ending in '-ar', e.g. 'comprar' (to buy) is conjugated in the present tense as
%'compro', 'compras', 'compra' in singular form (1st-3rd person)
%'compramos', 'compráis', 'compran' in plural (1st-3rd person)

$verb_ar$ = <V>:<> (\
  {<1sg>}:{o} |\
  {<2sg>}:{as} |\
  {<3sg>}:{a} |\
  {<1pl>}:{amos} |\
  {<2pl>}:{áis} |\
  {<3pl>}:{an} |\
  {<inf>}:{ar}) 


% class 3
%nouns that end in an unstressed vowel is added with 's' to form the plural
%e.g., la 'casa' (house) -> las 'casas' (houses)

$noun_vow$ = <N>:<> (\
  {<sg>}:{} |\
  {<pl>}:{s})


% class 4 (with complex rule)
%if a noun ends in 'ión', add 'es' and drop the accent over the 'o'
%e.g. la 'emoción' (emotion) -> las 'emociones' (emotions)

$noun_accent_rule$ = "noun_accent.lex" <N> (<sg>:<> | <pl>:{es})
$replace_accent$ = (ó:o) ^-> (__n<N>es)
$remove_N$ = (<N>:<>) ^-> ()
$noun_accent_rule$ = $noun_accent_rule$ || $replace_accent$ || $remove_N$


% class 5 (bonus rule)
%a noun takes a diminutive form to denote a smaller version of something
%e.g., el 'libro' (book) -> el 'librito' (booklet)
%or to emphasize endearment, e.g. la 'abuela' (grandmother) -> la 'abuelita' (granny)

$noun_fdim$ = <FemDimin>:<> <>:{ita}
$noun_mdim$ = <MascDimin>:<> <>:{ito}


% read lexica and remove word endings

$verbs$ = "verbs.lex" || [a-záéíóú]* {er}:{} | [a-záéíóú]* {ar}:{}
$verb_rules$ = $verb_er$ | $verb_ar$
$noun_fem$ = "nouns.lex" || [a-záéíóú]* {a}:{}
$noun_masc$ = "nouns.lex" || [a-záéíóú]* {o}:{}
$nouns$ = "nouns.lex" | $noun_fem$ $noun_fdim$ | $noun_masc$ $noun_mdim$ 

$MORPH$ = $verbs$ $verb_rules$ | $nouns$ $noun_vow$ | $noun_accent_rule$
$MORPH$