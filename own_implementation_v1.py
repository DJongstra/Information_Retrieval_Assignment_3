import hashlib
import random
import re
import string

from snapy import MinHash, LSH


def main():
    # example from https://pypi.org/project/snapy/
    documents = [
        'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
        'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
        'A helium atom has about four times as much mass as a hydrogen atom, so the composition changes when described as the proportion of mass contributed by different atoms.',
        'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
        'A helium atom has about four times as much mass as a hydrogen atom and the composition changes when described as a proportion of mass contributed by different atoms.',
        'Theoretical models indicate that if Jupiter had much more mass than it does at present, it would shrink.',
        'This process causes Jupiter to shrink by about 2 cm each year.',
        'Jupiter is mostly composed of hydrogen with a quarter of its mass being helium',
        'The Great Red Spot is large enough to accommodate Earth within its boundaries.'
    ]
    # changed 'large' to 'big' from document 8
    plagiarized_doc = "The Great Red Spot is big enough to accommodate Earth within its boundaries."

    for lsh in [
        LSHImplementation(n_gram=2, bands=20, rows=2),
        LSHLibrary(n_gram=2, bands=20, rows=2)
    ]:
        print(f"========== {lsh.__class__.__name__} ==========")
        s = 0.5
        lsh.add_documents(documents)
        lsh.compute()
        print(f"All similarities s>={s}")
        print(lsh.get_all_similarities(s=0.5))
        print(f"Find similar documents as plagiarized doc with s>={s}")
        print(lsh.query_content(plagiarized_doc, s=0.5))


# Basic functionality that the library as wel as our own implementation must handle
class LSHFunctionality:
    def __init__(self, n_gram, bands, rows, seed=123):
        self.n_gram = n_gram
        self.bands = bands
        self.rows = rows
        self.seed = seed
        self.signature_length = bands * rows
        self.original_documents = []

    # read directly from csv file
    def read_csv(self, csv_file):
        for _, row in csv_file.iterrows():
            self.original_documents.append(row['article'])

    # add documents as a list of strings
    def add_documents(self, documents: []):
        self.original_documents = documents

    # after adding documents, call compute to start the LSH
    def compute(self):
        raise Exception("virtual")

    # return all similarities >= s
    def get_all_similarities(self, s: float):
        raise Exception("virtual")

    # compare all docs to 'content' and return all where >= s
    def query_content(self, content: str, s: float):
        raise Exception("virtual")


# We first start with implementing this functionality with the library
# https://pypi.org/project/snapy/
class LSHLibrary(LSHFunctionality):
    def __init__(self, n_gram, bands, rows):
        super().__init__(n_gram, bands, rows)
        self.lsh = None

    def compute(self):
        self.lsh = LSH(
            MinHash(
                self.original_documents,
                n_gram=self.n_gram,
                n_gram_type='term',
                permutations=self.signature_length,
                seed=self.seed
            ),
            range(len(self.original_documents)),
            no_of_bands=self.bands
        )

    def get_all_similarities(self, s: float):
        return self.lsh.edge_list(min_jaccard=s, jaccard_weighted=True)

    # to query some content, we first have to add it to our set, minhash it and than query its id..
    def query_content(self, content: str, s: float):
        doc_id = len(self.original_documents)
        self.original_documents.append(content)

        # add to set (M)
        self.lsh.update(MinHash(
            [content],
            n_gram=self.n_gram,
            n_gram_type='term',
            permutations=self.signature_length,
            seed=self.seed
        ), [doc_id])

        # query matching documents
        return self.lsh.query(doc_id, min_jaccard=s)


# Our own implementation requires some additional methods to get the correct functionality
# We assume that after all documents are added, the set is static and no documents are added after.
# We can however compare new content to our existing set using query_content(doc, s)
class LSHImplementation(LSHFunctionality):
    def __init__(self, n_gram, bands, rows, seed=123):
        super().__init__(n_gram, bands, rows, seed)
        self.hash_tables = []
        self.M = []
        self.similarities = {}

        # prepare signature hash functions based on seed
        random.seed(seed)
        self.prepared_hash_functions = [HashFunction(random.getrandbits(64)) for _ in range(self.signature_length)]

    # "Should've used LSH, right?" -> "should have used lsh right"
    def preprocess_document(self, document: str):
        doc = document.lower()  # lower case
        doc = doc.replace("n't", " not").replace("'ve", " have").replace("'s", "")  # rewrite contractions
        doc = re.sub(" [^ ]*&amp[^ ]*", "", doc)  # remove random "&amp" in text
        doc = doc.translate(str.maketrans('', '', string.digits))  # remove numbers?
        doc = re.sub(" +", " ", doc)  # remove double spaces
        doc = doc.translate(str.maketrans('', '', string.punctuation))  # remove ALL punctuation
        return doc

    # -> "rose is a rose is a rose"
    # -> [["rose", "is", "a"], ["is", "a", "rose"], ["a", "rose", "is"], ["rose", "is", "a"], ["is", "a", "rose"]]
    # -> [44, 24, 17, 44, 24]
    # -> {44, 24, 17}
    def doc_to_hashed_shingles(self, doc):
        terms = doc.split()
        hash_f = HashFunction()  # key=0
        no_shingles = len(terms) - self.n_gram + 1
        return set([hash_f.compute_strings(terms[i:i + self.n_gram]) for i in range(no_shingles)])

    # Pre process the document, shingle its contents, hash the shingles and create the signature using minhash
    def doc_to_signature(self, original_doc):
        # "rose is a rose is a rose"
        doc = self.preprocess_document(original_doc)
        # To set of shingles: {34, 727, 1, .., 934}
        hashed_shingles = self.doc_to_hashed_shingles(doc)
        signature = []
        for hash_f in self.prepared_hash_functions:
            # returns shingle for which h_i outputs the lowest value
            min_hash = min(hashed_shingles, key=hash_f.compute_int64)
            signature.append(min_hash)
        return signature  # <- sketch!

    # Construct M, create hash tables and compute similarities
    def compute(self):
        self.M = self.construct_M()
        self.hash_tables = self.construct_hash_tables()
        self.similarities = self.construct_similarities()

    def construct_M(self):
        M = []
        for original_doc in self.original_documents:
            signature = self.doc_to_signature(original_doc)
            M.append(signature)
        return M

    # Construct a hash table (dictionary) for each band, the row values in the signature is a key in the table
    # If doc1 has values (1,2,3) for band 2, and doc2 also has values (1,2,3) for band 2,
    # then they will end up in the same bucket.
    def construct_hash_tables(self):
        bands_hash_tables = []
        for b in range(self.bands):
            hash_table = {}
            for doc_id in range(len(self.M)):
                signature = self.M[doc_id]
                key = tuple(signature[b * self.rows:(b + 1) * self.rows])
                if key in hash_table:
                    hash_table[key].append(doc_id)
                else:
                    hash_table[key] = [doc_id]
            bands_hash_tables.append(hash_table)
        return bands_hash_tables

    # Construct all similarities by keeping track of all hits between documents
    # Result -> {(doc1, doc2):5, (doc2, doc7):3}
    # If total_bands=10, then the jaccard for doc1&2 is 5/10 = 0.5
    def construct_similarities(self):
        similarities = {}
        for b in range(self.bands):
            for sim_list in self.hash_tables[b].values():
                no_docs = len(sim_list)
                if no_docs > 1:
                    for i in range(no_docs - 1):
                        for j in range(i + 1, no_docs):
                            key = tuple([sim_list[i], sim_list[j]])
                            if key in similarities:
                                similarities[key] += 1
                            else:
                                similarities[key] = 1
        return similarities

    # Get all document id's where the jaccard >= s
    def get_all_similarities(self, s: float):
        # Now the jaccard value is the amount of band hits / total_bands, but only return if >= s
        return [(doc1, doc2, hits / self.bands)
                for ((doc1, doc2), hits) in self.similarities.items() if hits / self.bands >= s]

    # Create a signature for the new document, and compare its bands with the bands hash table to find similar documents
    def query_content(self, content: str, s: float):
        similarities = {}
        signature = self.doc_to_signature(content)
        for b in range(self.bands):
            key = tuple(signature[b * self.rows:(b + 1) * self.rows])
            if key in self.hash_tables[b]:
                # all documents that share the same row values in band b
                for doc_id in self.hash_tables[b][key]:
                    # keep counters how many times another doc has the same band values
                    if doc_id in similarities:
                        similarities[doc_id] += 1
                    else:
                        similarities[doc_id] = 1

        # Now the jaccard value is the amount of band hits / total_bands, but only return if >= s
        return [(doc, hits / self.bands)
                for (doc, hits) in similarities.items() if hits / self.bands >= s]


# Hash function object that can be prepared
# If no key is provided this function will be a normal MD5 hash
# If two hash functions share the same key, they will also generate the same output for a certain input
class HashFunction:
    def __init__(self, key: int = None):
        self.key = key  # store key

    # ["rose", "is", "a"] -> 189939623769124324
    def compute_strings(self, shingle: []):
        h = hashlib.md5(self.int_to_bytes(self.key) if self.key else b'')
        for word in shingle:
            h.update(word.encode())
        return self.to_64_bit(h.digest())

    # (shingle hash) 189939623769124324 ->  (rank) 134237347983861913
    def compute_int64(self, shingle: int):
        h = hashlib.md5(self.int_to_bytes(self.key) if self.key else b'')
        h.update(self.int_to_bytes(shingle))
        return self.to_64_bit(h.digest())

    # convert 64 bit integer to 4 bytes
    def int_to_bytes(self, i: int):
        return int.to_bytes(i, length=8, byteorder='big', signed=False)

    # convert 16 byte hash digest (128 bit) to a 64 bit integer (6 bytes)
    def to_64_bit(self, md5_digest: bytes):
        return int.from_bytes(md5_digest[:8], byteorder='big', signed=False)


if __name__ == '__main__':
    main()
