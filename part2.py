from utils.es import get_term_vectors, get
from utils.text import stem_sentence

def format_to_liblinear(label, tf_dict):
    text = ""
    text += "{0} ".format(label)
    tf_set = []
    for feature, tf in tf_dict:
        tf_set.append("{0}:{1}".format(feature, tf))
    text += " ".join(tf_set) + "\n"
    return text

def build_features_dict():
    with open("./trec07p/full/index", "r") as doc_file:
        doc_list = doc_file.read().split("\n")

    docs_array = []
    terms = set()

    for doc in doc_list:
        try:
            label, doc = doc.split()
        except Exception as e:
            print(e)
            continue
        doc = doc.split("/").pop()
        docs_array.append(doc)

    for doc in docs_array:
        result = get(doc)
        is_test = result["_source"]["is_test"]
        if is_test == 0:
            results_dict = get_term_vectors(doc)
            if len(results_dict) > 0:
                results_dict = results_dict["text"]["terms"]
            else:
                print("Empty term vectors: " + doc)
                results_dict = {}

            for term in results_dict:
                terms.add(term)

    with open("features_list.txt", "w") as fl:
        for i, term in enumerate(terms):
            fl.write("{0} {1}\n".format(term, i + 1))

def read_features_dict():
    features_dict = {}
    with open("features_list.txt", "r") as fl:
        features_list = fl.read().split("\n")

    for f in features_list:
        try:
            term, term_id = f.split()
            features_dict[term] = term_id
        except Exception as e:
            print(e)

    return features_dict


def build_train_and_test_doc_dict():
    with open("./trec07p/full/index", "r") as doc_file:
        doc_list = doc_file.read().split("\n")

    docs_array = []
    train_set = set()
    test_set = set()

    for doc in doc_list:
        try:
            label, doc = doc.split()
        except Exception as e:
            print(e)
            continue
        doc = doc.split("/").pop()
        docs_array.append(doc)

    for doc in docs_array:
        result = get(doc)
        is_test = result["_source"]["is_test"]
        if is_test == 0:
            train_set.add(result["_id"])
        else:
            test_set.add(result["_id"])

    with open("train_list.txt", "w") as trl:
        for i, doc in enumerate(train_set):
            trl.write("{0} {1}\n".format(doc, i + 1))

    with open("test_list.txt", "w") as tl:
        for i, doc in enumerate(test_set):
            tl.write("{0} {1}\n".format(doc, i + 1))

def read_train_and_test_dict():
    train_dict = []
    test_dict = []
    with open("./train_list.txt", "r") as fl:
        train_list = fl.read().split("\n")

    for f in train_list:
        try:
            doc, doc_pos = f.split()
            train_dict.append(doc)
        except Exception as e:
            print(e)

    with open("./test_list.txt", "r") as fl:
        test_list = fl.read().split("\n")

    for f in test_list:
        try:
            doc, doc_pos = f.split()
            test_dict.append(doc)
        except Exception as e:
            print(e)

    return train_dict, test_dict

def build_sparse_matrix():
    train_matrix_file = open("train_matrix.txt", "a")
    test_matrix_file = open("test_matrix.txt", "a")

    features_dict = read_features_dict()
    train_dict, test_dict = read_train_and_test_dict()

    for doc in train_dict:
        result = get(doc)
        is_spam = result["_source"]["is_spam"]

        results_dict = get_term_vectors(doc)
        if len(results_dict) > 0:
            results_dict = results_dict["text"]["terms"]
            tf_tuples = []

            for term in results_dict:
                term_id = features_dict[term]

                tf_tuples.append((term_id, results_dict[term]['term_freq']))

            tf_tuples = sorted(tf_tuples, key=lambda tup: int(tup[0]))
            train_matrix_file.write(format_to_liblinear(is_spam, tf_tuples))

        else:
            print("Empty term vectors: " + doc)
            continue

    for doc in test_dict:
        result = get(doc)
        is_spam = result["_source"]["is_spam"]

        results_dict = get_term_vectors(doc)

        if len(results_dict) > 0:
            results_dict = results_dict["text"]["terms"]
            tf_tuples = []

            for term in results_dict:
                if term in features_dict:
                    term_id = features_dict[term]
                    tf_tuples.append((term_id, results_dict[term]['term_freq']))

            if len(tf_tuples) > 0:
                tf_tuples = sorted(tf_tuples, key=lambda tup: int(tup[0]))
                test_matrix_file.write(format_to_liblinear(is_spam, tf_tuples))
            else:
                print("No term matched training: {0}".format(doc))
                continue

        else:
            print("Empty term vectors: " + doc)
            continue
    train_matrix_file.close()
    test_matrix_file.close()


# build_features_dict()
# build_train_and_test_doc_dict()
build_sparse_matrix()
