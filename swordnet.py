'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SWordNet description;;;
;;;;;;;;;;;;;;;;;;;;;;;

	Here a class is implemented for SWordNet. The SWordNet technique extracts
	semantically similar word/phrase pairs from software source code.  The
	relationship between words and phrases is not defined. That is, whether the
	words or phrases are synonyms, hypernym hyponyms, meronyms, antonyms etc is not
	defined using this method.

Steps to form pairs;;;
;;;;;;;;;;;;;;;;;;;;;;

    (1) separate comment and code from the source code
    (2) extract identifier names from code and sentences from commment
    (3) tokenize identifiers and sentences
    (4) cluster identifiers and sentences based on words
    (5) perform longest common subsequence (lcs) algorithm to find similarity between sentences/identifiers
    (6) if similar (similarity>threshold) then extract pairs from the sequences (#common words/length of short sequence)
    (7) remove pairs having stopwords and stem to remove pairs with same roots

Example;;;
;;;;;;;;;;

    Sentences: "Mask all interrupt sources", "Disable all irq sources"
    Tokenize: <mask, all, interrupt, sources> , <disable, all, irq, sources>
    LCS: <all, sources>
    Similarity: 2/4 = 0.5
    Pairs: <mask, disable>, <interrupt, irq>

Inspired from publication;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    Jinqui Yang, and Lin Tan, "SWordNet: Inferring semantically related words from software context"
    Empirical Software Engineering, 2014, Volume 19, Number 6, Page 1856

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

__author__ = ["Shayan Ali Akbar"]
__email__ =  ["sakbar@purdue.edu"]

import os
import regex
import string

from reader import *
from preprocessor import *

from itertools import groupby, count

class SWordNet:

    def __init__(self):
        # data related
        self.data_path = None
        self.file_list = None
        self.content = None
        self.project_name = None
        self.output_path = None

        # global tokenized comment and code lists
        self.comments = []
        self.codes = []

        # compiled regexes
        self.comment_regex1 = regex.compile(r'/\*(?:.|[\r\n])*?\*/', regex.MULTILINE)
        self.comment_regex2 = regex.compile(r'//(?:.|[\r\n])*?\n')
        self.segment_sentence_regex = regex.compile(r'\s|\!|\?}\;|\.|\n')

        # stopwords related
        self.stopwords_file = None
        self.list_stopwords = None

        # parameters
        self.shortest = None
        self.longest = None
        self.gap = None
        self.threshold = None

        # final_pairs
        self.pairs_comment_comment = {}
        self.pairs_code_code = {}
        self.pairs_comment_code = {}

    def remove_punctuations(self, content):
        '''

		Remove all punctuations from the contents

        :param content: raw content
        :return: processed content
        '''

        return "".join(l if l not in string.punctuation else " " for l in content)

    def remove_punctuations_except_underscore(self, content):
        '''

		Remove all punctuations from the contents except underscore

        :param content: raw content
        :return: the processed content
        '''

        puncs = "!@#$%^&*()[]{};:,./<>?\|`~-=_+\""
        return "".join(l if l not in puncs else " " for l in content)

    def remove_stopwords(self, content):
        '''

        Remove all stopwords from the content

        :param content: raw content
        :return: the processed content
        '''

        for stopword in self.list_stopwords:
            pattern = " " + stopword + " "
            content = regex.sub(pattern, " ", content)

        content = ''.join([i for i in content if not i.isdigit()])
        return content

    def perform_camel_case_splitting(self, content):
        '''

        Convert all camelcase terms into individual terms

        :param content: raw content
        :return: processed content without any camelcase terms
        '''

        matches = regex.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', content)
        return " ".join([m.group(0) for m in matches])

    def read_stopwords(self):
        '''

        Read stopwords from stopwords_file

        :return: list of stopwords
        '''

        list_stopwords = []
        with open(self.stopwords_file) as f:
            for line in f:
                list_stopwords.append(line.strip())
        self.list_stopwords = list_stopwords
        return list_stopwords

    def separate_comment_code(self):
        '''

        Separate comment from code and make separate strings

        :return: comment only text, code only text
        '''

        comment = []
        for x in regex.findall(self.comment_regex1, self.content):
            comment.append(x)
        for x in regex.findall(self.comment_regex2, self.content):
            comment.append(x)
        comment = '\n'.join(set(comment))

        code = regex.sub(self.comment_regex2, '', regex.sub(self.comment_regex1, '', self.content))
        return comment, code

    def segment_comment_sentence(self, comment):
        '''

        Separate sentences in the comment

        :param comment: comment text
        :return: list of sentences in comment
        '''

        comment_sentences = []
        for x in (regex.split(self.segment_sentence_regex, comment)):
            comment_sentences.append(x)
        return comment_sentences

    def segment_code_identifier(self, code):
        '''

		Separate identifiers in the code

        :param code: code text
        :return: list of identifiers in code
        '''

        return regex.split(self.segment_sentence_regex, code)

    def cluster_based_on_term_comments_codes(self, codes_comments):
        '''

        :param codes_comments:
        :return:
        '''

        terms = set([item for sublist in codes_comments for item in sublist[1:]])
        cluster_comments = []
        cluster_terms = []
        for term in terms:
            if len(codes_comments) == 0: break
            comments_with_term = [c for c in codes_comments if term in c[1:]]
            if len(comments_with_term) > 1:
                cluster_comments.append(comments_with_term)
                cluster_terms.append(term)
                comments = [c for c in codes_comments if term not in c[1:]]
        return cluster_comments, cluster_terms

    def cluster_based_on_term(self, comments):
        '''

        Cluster the text/comments/code identifiers based on terms

        :param comments: list of texts
        :return: clustered texts (list of list), terms based on which clusters are made
        '''

        terms = set([item for sublist in comments for item in sublist])
        cluster_comments = []
        cluster_terms = []
        for term in terms:
            if len(comments) == 0: break
            comments_with_term = [c for c in comments if term in c]
            if len(comments_with_term) > 1:
                cluster_comments.append(comments_with_term)
                cluster_terms.append(term)
                comments = [c for c in comments if term not in c]
        return cluster_comments, cluster_terms

    def separate_comment_code_in_cluster(self, comment_code):
        '''

        Separate comments and codes. Comments should have a '1' as their first term, codes '2'

        :param comment_code: list of comment and code text with 1 and 2 as their first terms respectively
        :return: comment text, code text
        '''

        code = [v for v in comment_code if v[0] == '1']
        comment = [v for v in comment_code if v[0] == '2']
        return comment, code

    def lcs(self, a, b):
        '''

        Takes as input two sequences and perform longest common subsequence (lcs) algorithm on it

        :param a: first sequence
        :param b: second sequence
        :return: length of lcs, the lcs string, positions at which first sequence has lcs terms,
                 positions at which second sequence has lcs terms, positions at which first sequence
                 has terms different from lcs terms, positions at which second sequence has terms
                 different from lcs terms
        '''

        lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        x, y = len(a), len(b)
        a_common_seq_pos = []
        b_common_seq_pos = []
        a_diff_seq_pos = []
        b_diff_seq_pos = []
        while x != 0 and y != 0:
            # print ('****')

            if lengths[x][y] == lengths[x - 1][y]:
                a_diff_seq_pos = [x - 1] + a_diff_seq_pos
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                b_diff_seq_pos = [y - 1] + b_diff_seq_pos
                y -= 1
            else:
                assert a[x - 1] == b[y - 1]
                result = [a[x - 1]] + result
                a_common_seq_pos = [x - 1] + a_common_seq_pos
                b_common_seq_pos = [y - 1] + b_common_seq_pos

                x -= 1
                y -= 1

        return len(result), result, a_common_seq_pos, b_common_seq_pos, a_diff_seq_pos, b_diff_seq_pos

    def find_new_similarity_measure(self, len_common_subseq, X, Y):
        '''

        Find similarity measure based on idf values

        :param len_common_subseq: length of longest common subsequence
        :param X: first sequence
        :param Y: second sequence
        :return: similarity score
        '''

        print

    def find_similarity_measure(self, len_common_subseq, X, Y):
        '''

        Finds out how similar two sequences are based on lcs.
        Two methods are there for finding similarity
        One based on simply dividing lcs length by length of shorter sequence
        Another based on idf values (new similarity measure)

        :param len_common_subseq: length of lcs
        :param X: first sequence
        :param Y: second sequence
        :return: similarity score
        '''

        new_simi_measure = False
        if not new_simi_measure:
            return float(len_common_subseq) / float(min(len(X), len(Y)))
        else:
            return self.find_new_similarity_measure(len_common_subseq, X, Y)


    def extract_pairs(self, a, b, a_diff_pos, b_diff_pos):
        '''

        Once we have run the lcs algorithm on the sequences, we get all the poisitons at which
        the two sequences differ. We can use this information to extract the word or phrase pairs
        that are related. The algorithm to do that works as follows:
            cluster the list of positions in sequence a at which the term differ from the lcs terms
            cluster the list of positions in sequence b at which the term differ from the lcs terms
            for each cluster in a find the first cluster in b which has the first term before and first term after clusters
            if the first term before and first term after cluster is the same for the two clusters then
            we have the pair as the cluster of a and cluster of b

        :param a: sequence a
        :param b: sequence b
        :param a_diff_pos: positions at which a differ from lcs terms
        :param b_diff_pos: positions at which b differ from lcs terms
        :return: pairs formed from these two subsequences a and b
        '''

        clusts_pos1 = [list(g) for _, g in groupby(a_diff_pos, lambda n, c=count(): n - next(c))]
        clusts_pos2 = [list(g) for _, g in groupby(b_diff_pos, lambda n, c=count(): n - next(c))]

        pairs = {}
        for c_p_1 in clusts_pos1:
            if c_p_1[0] - 1 < 0 or c_p_1[-1] + 1 > len(a) - 1:
                continue
            tmp1 = (a[c_p_1[0] - 1], a[c_p_1[-1] + 1])

            for c_p_2 in clusts_pos2:
                if c_p_2[0] - 1 < 0 or c_p_2[-1] + 1 > len(b) - 1:
                    continue
                tmp2 = (b[c_p_2[0] - 1], b[c_p_2[-1] + 1])

                if tmp1 == tmp2:
                    x = ' '.join([a[v] for v in c_p_1])
                    y = ' '.join([b[v] for v in c_p_2])
                    if (x, y) in pairs:
                        pairs[(x, y)][0] += 1
                        pairs[(x, y)].extend([" ".join(a), " ".join(b)])
                    else:
                        pairs[(x, y)] = [1, " ".join(a), " ".join(b)]

        return pairs

    def find_comment_comment_pairs(self):
        '''

        Loop over all the comment pairs and form pairs
        Write pairs in a file

        :return: pairs extracted
        '''

        self.shortest = 4
        self.longest = 10
        self.gap = 3
        self.threshold = 0.7

        comments = self.comments
        comments = [comment for comment in comments if len(comment) >= self.shortest]
        comments = [comment for comment in comments if len(comment) <= self.longest]

        cluster_comments, cluster_terms = self.cluster_based_on_term(comments)

        counter = 0
        pairs = {}
        for cluster, term in zip(cluster_comments, cluster_terms):
            if counter %1000 == 0: print (counter, "/", len(cluster_comments))

            for i in range(0, len(cluster) - 1):
                for j in range(i, len(cluster)):
                    if abs(len(cluster[i]) - len(cluster[j])) < self.gap:
                        a = cluster[i]
                        b = cluster[j]
                        len_result, result, a_pos, b_pos, a_diff_pos, b_diff_pos = self.lcs(a, b)
                        simi_measure = self.find_similarity_measure(len_result, a, b)

                        if simi_measure > self.threshold and simi_measure < 1.0:
                            pair = self.extract_pairs(a, b, a_diff_pos, b_diff_pos)
                            if len(pair) > 0:
                                for k, v in pair.items():
                                    if k in pairs:
                                        pairs[k][0] += v[0]
                                        pairs[k].extend(v[1:])
                                    elif (k[1], k[0]) in pairs:
                                        pairs[(k[1], k[0])][0] += v[0]
                                        pairs[(k[1], k[0])].extend(v[1:])
                                    else:
                                        pairs[k] = v

            counter += 1

        if len(pairs) > 0:
            for p, v in pairs.items():
                num_distinct_context = len(set(v[1:])) / 2
                v = [num_distinct_context] + v
                pairs[p] = v

        print ("Got all the pairs. Now writing them on the disk")
        with open(self.output_path + self.project_name + '/comment_comment_pairs.csv', 'w') as f:
            for p, v in pairs.items():
                f.write(','.join(p) + ',' + ','.join([str(e) for e in v]) + '\n')

        self.pairs_comment_comment = pairs
        return pairs

    def find_code_code_pairs(self):
        '''

        Loop over all the code pairs and form pairs
        Write pairs in a file

        :return: pairs extracted
        '''

        self.shortest = 4
        self.longest = 6
        self.gap = 0
        self.threshold = 0.55

        codes = self.codes
        codes = [code for code in codes if len(code) >= self.shortest]
        codes = [code for code in codes if len(code) <= self.longest]

        cluster_codes, cluster_terms = self.cluster_based_on_term(codes)

        counter = 0
        pairs = {}
        for cluster, term in zip(cluster_codes, cluster_terms):
            if counter %1000 == 0: print (counter, "/", len(cluster_codes))
            for i in range(0, len(cluster) - 1):
                for j in range(i, len(cluster)):
                    if abs(len(cluster[i]) - len(cluster[j])) <= self.gap:
                        a = cluster[i]
                        b = cluster[j]
                        len_result, result, a_pos, b_pos, a_diff_pos, b_diff_pos = self.lcs(a, b)
                        simi_measure = self.find_similarity_measure(len_result, a, b)

                        if simi_measure > self.threshold and simi_measure < 1.0:
                            pair = self.extract_pairs(a, b, a_diff_pos, b_diff_pos)
                            if len(pair) > 0:
                                for k, v in pair.items():
                                    if k in pairs:
                                        pairs[k][0] += v[0]
                                        pairs[k].extend(v[1:])
                                    elif (k[1], k[0]) in pairs:
                                        pairs[(k[1], k[0])][0] += v[0]
                                        pairs[(k[1], k[0])].extend(v[1:])
                                    else:
                                        pairs[k] = v

            counter += 1

        if len(pairs) > 0:
            for p, v in pairs.items():
                num_distinct_context = len(set(v[1:])) / 2
                v = [num_distinct_context] + v
                pairs[p] = v

        print ("Got all the pairs. Now writing them on the disk")
        with open(self.output_path + self.project_name + '/code_code_pairs.csv', 'w') as f:
            for p, v in pairs.items():
                f.write(','.join(p) + ',' + ','.join([str(e) for e in v]) + '\n')

        self.pairs_code_code = pairs
        return pairs

    def find_comment_code_pairs(self):
        '''

        Loop over all the comment-code pairs and form pairs
        Write pairs in a file

        :return: pairs extracted
        '''

        self.shortest = 4
        self.longest = 6
        self.gap = 1
        self.threshold = 0.65

        codes = self.codes
        codes = [code for code in codes if len(code) >= self.shortest]
        codes = [['1'] + code for code in codes if len(code) <= self.longest]

        comments = self.comments
        comments = [comment for comment in comments if len(comment) >= self.shortest]
        comments = [['2'] + comment for comment in comments if len(comment) <= self.longest]

        cluster_comments_codes, cluster_terms = self.cluster_based_on_term_comments_codes(codes + comments)

        counter = 0
        pairs = {}
        for cluster_comment_code, term_comment_code in zip(cluster_comments_codes, cluster_terms):
            if counter %1000 == 0: print (counter, "/", len(cluster_comments_codes))

            cluster_comment, cluster_code = self.separate_comment_code_in_cluster(cluster_comment_code)

            for i in range(0, len(cluster_comment)):
                for j in range(0, len(cluster_code)):
                    if abs(len(cluster_comment[i]) - len(cluster_code[j])) <= self.gap:
                        a = cluster_comment[i][1:]
                        b = cluster_code[j][1:]
                        len_result, result, a_pos, b_pos, a_diff_pos, b_diff_pos = self.lcs(a, b)
                        simi_measure = self.find_similarity_measure(len_result, a, b)

                        if simi_measure > self.threshold and simi_measure < 1.0:
                            pair = self.extract_pairs(a, b, a_diff_pos, b_diff_pos)
                            if len(pair) > 0:
                                for k, v in pair.items():
                                    if k in pairs:
                                        pairs[k][0] += v[0]
                                        pairs[k].extend(v[1:])
                                    elif (k[1], k[0]) in pairs:
                                        pairs[(k[1], k[0])][0] += v[0]
                                        pairs[(k[1], k[0])].extend(v[1:])
                                    else:
                                        pairs[k] = v

            counter += 1

        if len(pairs) > 0:
            for p, v in pairs.items():
                num_distinct_context = len(set(v[1:])) / 2
                v = [num_distinct_context] + v
                pairs[p] = v

        print ("Got all the pairs. Now writing them on the disk")
        with open(self.output_path + self.project_name + '/comment_code_pairs.csv', 'w') as f:
            for p, v in pairs.items():
                f.write(','.join(p) + ',' + ','.join([str(e) for e in v]) + '\n')

        self.pairs_comment_code = pairs
        return pairs

    def find_pairs(self):
        '''

        Find pairs using swordnet technique

        :return: comment-comment comment-code and code-code pairs
        '''

        reader = Reader()
        reader.data_path = self.data_path
        extension = "java"
        reader.pattern = "*." + extension
        reader.get_file_list()
        print("\n")
        print("Got list of files from the data directory")
        print("To view these filenames open etc/filenames.txt")

        self.stopwords_file = 'etc/stopword-list.txt'
        self.read_stopwords()

        files = reader.files
        counter_files = 0
        print ("\n")
        print ("Extracting comments and code identifiers from files")
        for file in files:
            if counter_files % 100 == 0: print(counter_files, '/', len(files))
            self.content = reader.read_file(file)

            comment, code = self.separate_comment_code()
            comment_sentences = self.segment_comment_sentence(comment)
            code_blocks = self.segment_code_identifier(code)

            comment_sentences = set([self.remove_punctuations_except_underscore(c_s).strip() for c_s in comment_sentences])
            code_identifiers = set([self.remove_punctuations(c_i).strip() for c_i in code_blocks])
            comment_sentences = [self.perform_camel_case_splitting(c_s) for c_s in comment_sentences]
            code_identifiers = [self.perform_camel_case_splitting(c_i) for c_i in code_identifiers]

            self.comments.extend([list(filter(None, regex.split('\s', x.lower()))) for x in comment_sentences])
            self.codes.extend([list(filter(None, regex.split('\s', x.lower()))) for x in code_identifiers])

            counter_files += 1

        if not os.path.exists(self.output_path + self.project_name): os.makedirs(self.output_path + self.project_name)

        print ("\n")
        print ("Finding comment comment pairs")
        self.find_comment_comment_pairs()
        print ('Found all comment comment pairs')

        print ("\n")
        print ("Finding code code pairs")
        self.find_code_code_pairs()
        print ('Found all code code pairs')

        print ("\n")
        print ("Finding comment code pairs")
        self.find_comment_code_pairs()
        print ('Found all comment code pairs')
        print ("\n\n\n")

        return self.pairs_comment_comment, self.pairs_comment_code, self.pairs_code_code
