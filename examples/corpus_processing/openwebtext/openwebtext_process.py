import os
import sys
import ray
import json
import tqdm
import spacy
from typing import List

from utils import merge_short_sents

TEMP_PATH = "TEMP"

ray.init(num_cpus=64, memory=75000000000)


@ray.remote
def count_files(path):
    count = len(os.listdir(path))
    return count


objects = []

for file_num in range(len(os.listdir(TEMP_PATH))):
    obj_ids = count_files.remote(os.path.join(TEMP_PATH, str(file_num)))
    objects.append(obj_ids)

total = 0
for i in range(len(os.listdir(TEMP_PATH))):
    total += ray.get(objects[i])

#count total number of files
print(f"total num of files: {total}")

# ## Get all file names

# In[6]:


@ray.remote
def get_filenames(path):
    filenames = os.listdir(path)
    filenames = [os.path.join(path, filename) for filename in filenames]
    return filenames


# In[7]:

all_obj_ids = []

for file_num in range(len(os.listdir(TEMP_PATH))):
    obj_id = get_filenames.remote(os.path.join(TEMP_PATH, str(file_num)))
    all_obj_ids.append(obj_id)

all_filenames = []

print("retrieving all filenames")

for i in tqdm.trange(len(os.listdir(TEMP_PATH))):
    all_filenames.extend(ray.get(all_obj_ids[i]))

# ## Write into DATA

# initialize SpaCy
spacy.prefer_gpu()
# we only need sentencizer
nlp = spacy.load(
    "en",
    disable=[
        "ner", "tagger", "parser", "merge_noun_chunks", "merge_entities", "merge_subtokens", "entity_ruler", "textcat",
        "entity_linker"
    ]
)
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# single case
filename = all_filenames[0]

with open(filename, "r") as f_read:
    data = f_read.read()

spacy_doc = nlp(data)
doc_sents = [sent.text for sent in spacy_doc.sents]
doc_sents = merge_short_sents(doc_sents)

# In[11]:

print("Processed Document Example:")
print(doc_sents)

# In[1]:


def process_document(filename: str) -> List[str]:
    with open(filename, "r") as f_read:
        data = f_read.read()

    spacy_doc = nlp(data)
    doc_sents = [sent.text for sent in spacy_doc.sents]
    doc_sents = merge_short_sents(doc_sents)

    return doc_sents


@ray.remote
def write_data(filenames: List[str], file_descripts: List[str] = None, output_name=None, sector_id=None):
    ""
    f_write = open(output_name, "w")
    # write file descriptions
    f_write.write(json.dumps(file_descripts))
    f_write.write("\n")

    if sector_id == 0:
        pbar = tqdm.tqdm(filenames)
    else:
        pbar = filenames

    for filename in pbar:

        doc_sents = process_document(filename)
        # dump into json
        doc_sents = json.dumps(doc_sents)

        # write into jsonl
        f_write.write(doc_sents)
        f_write.write("\n")

    f_write.close()

    return 0


# define the sector size
sector_size = 2**17

remote_objs = []

for sector_id, sector_start_pos in enumerate(range(0, len(all_filenames), sector_size)):
    sector_filenames = all_filenames[sector_start_pos:sector_start_pos + sector_size]
    sector_file_descripts = ["openwebtext_" + str(i) for i in range(len(sector_filenames))]

    obj_id = write_data.remote(
        sector_filenames, sector_file_descripts, os.path.join("DATA", f"{sector_id}.jsonl"), sector_id
    )
    remote_objs.append(obj_id)

for sector_id, sector_start_pos in enumerate(range(0, len(all_filenames), sector_size)):
    assert ray.get(remote_objs[sector_id]) == 0
