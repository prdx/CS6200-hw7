from utils.es import get_term_vectors, get
from utils.text import stem_sentence

labels_dict = {}

def build_matrix():
    # Build the base first
    matrix = {}
    with open("./trec07p/full/index", "r") as doc_file:
        doc_list = doc_file.read().split("\n")

    for doc in doc_list:
        try:
            label, doc = doc.split()
        except Exception as e:
            print(e)
            continue
        doc = doc.split("/").pop()
        labels_dict[doc] = 1 if label == "spam" else 0
        matrix[doc] = []

    with open("./custom_features.txt", "r") as features_file:
        features_list = features_file.read().split("\n")
        features_list.pop()

    text_empty = False
    for doc in matrix:
        results_dict = get_term_vectors(doc)
        if len(results_dict) > 0:
            results_dict = results_dict["text"]["terms"]
        else:
            print("Empty term vectors: " + doc)
            results_dict = {}

        for feature in features_list:
            word = stem_sentence(feature)
            try:
                matrix[doc].append(str(results_dict[word]['term_freq']))
            except:
                matrix[doc].append('0')

        # if text_empty == True:
            # matrix[doc].append(0)

        # # Append label
        # matrix[doc].append(labels_dict[doc])

    return matrix

def build_matrix_training_test(matrix):
    matrix_train = {}
    matrix_test = {}
    for doc in matrix:
        # Get whether it is training or not
        result = get(doc)
        is_test = result["_source"]["is_test"]
        is_spam = result["_source"]["is_spam"]
        doc_length = result["_source"]["doc_length"]
        if is_test == 0:
            matrix_train[doc] = matrix[doc]
            matrix_train[doc].append(str(doc_length))
            matrix_train[doc].append(str(is_spam))
        else:
            matrix_test[doc] = matrix[doc]
            matrix_test[doc].append(str(doc_length))
            matrix_test[doc].append(str(is_spam))

    return matrix_train, matrix_test


matrix = build_matrix()
matrix_train, matrix_test = build_matrix_training_test(matrix)


with open("matrix_train.csv", "w") as train, open("matrix_test.csv", "w") as test:
    for doc in matrix:
        string = "{0},{1}\n".format(doc, ",".join(matrix[doc]))
        if doc in matrix_train:
            train.write(string)
        else:
            test.write(string)


