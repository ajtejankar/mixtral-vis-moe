from itertools import chain
import os
from collections import Counter, defaultdict

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "stem": ["math", "engineering", "computer science", "physics", "chemistry", "biology"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["geography", "politics", "culture", "economics", "psychology"],
    "other": ["other", "business", "health"],
}


with open('subjects.txt', 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

subcat_to_subj = defaultdict(list)
for i, (subj, subcat) in enumerate(subcategories.items()):
    subcat_to_subj[subcat[0]].append(i)
subj_to_subcat = {subjects[v]: k for k, vv in subcat_to_subj.items() for v in vv}

cat_to_subj = defaultdict(list)
for i, (cat, subcats) in enumerate(categories.items()):
    cat_name = cat.split()[0].lower().replace('social', 'social sciences')
    for subcat in subcats:
        cat_to_subj[cat_name].extend(subcat_to_subj[subcat])
subj_to_cat = {subjects[v]: k for k, vv in cat_to_subj.items() for v in vv}

cat_list = ['stem', 'humanities', 'social sciences', 'other']
subcat_list = list(chain(*[categories[c] for c in cat_list]))

