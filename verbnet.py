'''
VerbNet description;;;
;;;;;;;;;;;;;;;;;;;;;;

    Here a class is implemented for VerbNet. The VerbNet technique extracts semantically
    similar verb pairs from software source code specifically using the method name
    and the leading comment to the method declaration.


Steps to form pairs;;;
;;;;;;;;;;;;;;;;;;;;;;

    (1) Identifying descriptive leading comment
            Eliminate leading comment blocks that are indicative of non-descriptive comments.
                . If comment block starts with all caps TODO HACK FIX it is discarded
                . If comment block starts with a personal pronoun like We I You it is discarded
            Divide each comment block into separate phrases
                . The phrases are segmented using splitting based on . ; :
            For each phrase in the comment block decide if it is a descriptive phrase
                . If the phrase starts with an IF clause it is discarded
                . If the phrase contains any of the verb phrases returns, called from, performed by, performed when, uses, see it is discarded
                . If the phrase contains keywords of programming language like overrides implements it is discarded
                . If the phrase contains past tense like added it is discarded
    (2) Extracting main verb from leading comment
            Perform Stanford POS tagging on the phrase to identify verbs
            Check if non-verb tagged words in the phrase are tagged as verbs in WordNet. If yes mask them as verb.
            Resolve made-up verbs (CURRENTLY NOT IMPLEMENTED ,,, seen that words like intro become tro marked as verb which are sort of false positive as tro is not really a verb or a word even)
                If a non-verb tagged word contains prefix or suffix remove them (re dis en ex out up un im in mis pre able en er ful ible ion less ly ness)
                For the resulting WORD put it in all the following templates
                    This method will WORD something
                    This method will WORD
                    This method WORD something
                Pass each of the above templates to the Stanford POS tagger and if the word is tagged as verb then mark it as verb
            Loop over each tagged verb in sequence and filter out based on following rules. The first verb that survives all rules wins.
                . Too frequent verbs like do, tries, test, perform, determine, is, are, was, has are ignored.
                . Verbs in past participle form are filtered out like given
                . Verbs should appear within first three words in phrase or is preceded by a 'to'
                . Verbs in third person singular form should not appear after first three words
                . Gerund (-ing) survives only if it is preceded by a preposition
    (3) Extracting main verb from the method name
            Use a specialized POS tagger for tagging words in method names (Gupta POS tagging of method names [21] in MSR paper)
            Loop over each word tagged as verb from the POS tagger to filter out using following rules:
                . If the verb is in past or past participle tense then it is discarded
                . Linking verbs like is am are has can are discarded
                . Method names starting with third person singular are ignored
                . Gerunds followed by a past tense verb are ignored
                . Method names starting with third person singular verbs are ignored (VP::3PS)
                . If the verb is in past or past participle tense then it is discarded (VP::pp,VP::pastV)
                . Linking verbs like is am are has can are discarded (VP::irV)
                . Gerunds followed by a past tense verb are ignored (VP::ingV)
    (4) Form the verb pair if they are different and keep a count of how many times the verb pair is encountered



Example;;;
;;;;;;;;;;

    Method and leading comment:     "jsc_functionAbort", "Cancels the HTTP request"
    Is leading comment descriptive: Yes, proceed!
    Verbs from method name:         abort
    Main verb in method name:       abort
    Verbs from comment:             cancels
    Main verb in comment:           cancels
    Verb pair:                      <abort, cancels>


'''

__author__ = ["Shayan Ali Akbar"]
__email__ =  ["sakbar@purdue.edu"]

import os
import nltk
import regex
import string

from reader import *
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger as stanfpos

from itertools import groupby, count
from itertools import islice

nltk.internals.config_java("/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java")

class VerbNet:

    def __init__(self):

        # data related
        self.data_path = None
        self.file_list = None
        self.content = None
        self.project_name = None
        self.output_path = None

        # compiled regexes
        self.method_comment_regex = regex.compile(r'(/\*(?:.|[\r\n])*?\*/)\s*(\n+)\s*(public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *(\([^\)]*\)) *(\{?|[^;])', regex.MULTILINE)

        # rule related
        self.personal_pronouns = ['i','you','we','they','he','she']
        self.context_verbs = ['param', 'return', 'returns', 'perform', 'performs', 'performed', 'use', 'uses', 'used', 'see', 'call', 'calls', 'called']
        self.java_keyverbs = ['overrides', 'override', 'implement','implements',
                              'catch', 'catches', 'extends', 'extend',
                              'interface', 'interfaces', 'throw', 'throws']

        self.jar = 'stanford-postagger-2018-02-27/stanford-postagger.jar'
        self.posmodel = 'stanford-postagger-2018-02-27/models/english-left3words-distsim.tagger'
        self.stanf_pos_tagger = stanfpos(self.posmodel, self.jar, encoding='utf8')

        self.prefix = ['re', 'dis', 'ex', 'out', 'up', 'un', 'im', 'in', 'mis', 'pre']
        self.suffix = ['able', 'ible', 'less', 'ness', 'en']

        self.too_frequent_verbs = ['do', 'does', 'try', 'tries', 'test', 'tests', 'performs', 'perform', 'determine', 'determines',
                                  'is', 'are', 'was', 'has']

        # final_pairs
        self.pairs = {}

    def remove_punctuations(self, content):
        '''

		Remove all punctuations from the contents

        :param content: raw content
        :return: processed content
        '''

        return "".join(l if l not in string.punctuation else " " for l in content)

    def perform_camel_case_splitting(self, content):
        '''
		Convert all camelcase terms into individual terms
		ret1: processed content without any camelcase terms
		'''

        matches = regex.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', content)
        return " ".join([m.group(0) for m in matches])

    def identify_desc_lead_comment(self, comment_block):

        comment_block = comment_block.replace("*","").strip()
        comment_block = comment_block.replace("/","")

        # check if comment block can be discarded
        first_word = comment_block.split(' ', 1)[0]
        if first_word.isupper() and len(first_word)>1:
            return False
        if first_word.lower() in self.personal_pronouns:
            return False
        if first_word.lower() == '(non-javadoc)':
            return False

        # divide block into phrases
        phrases = regex.split('[;:.]', comment_block)

        # filter phrases that are nondescriptive
        all_phrases = []
        for phrase in phrases:
            phrase = phrase.lower().strip()
            phrase = self.remove_punctuations(phrase)

            if phrase.split(' ', 1)[0] == "if":
                continue

            if any(x in phrase for x in self.context_verbs):
                continue

            if any(x in phrase for x in self.java_keyverbs):
                continue

            phrase = phrase.split()

            if len(phrase) < 3:
                continue

            all_phrases.append(phrase)

        if len(all_phrases) == 0: return False
        return all_phrases

    def remove_prefix_suffix(self, word):

        for x in self.prefix:
            if word.startswith(x):
                word = regex.sub(x, '', word)
                return word
        for x in self.suffix:
            if word.endswith(x):
                word = regex.sub(x, '', word)
                return word
        return word

    def get_main_verb_from_comment(self, verbs_names):

        verbs_names_new = []

        for vs_ns in verbs_names:
            for v_n in vs_ns[1:]:

                if v_n[0][0] in self.too_frequent_verbs:
                    continue

                if v_n[0][1] == 'VBN':
                    continue

                if v_n[2] > 3:
                    if v_n[3][0] != 'to':
                        continue

                if v_n[0][1] == 'VBZ' and v_n[2] > 3:
                    continue

                if v_n[0][1] == 'VBG' and v_n[3][1] != 'IN':
                    continue

                verbs_names_new.append([v_n,vs_ns[0]])
                break

        return verbs_names_new

    def prepare_method_input_line(self,name):
        name = [' '.join(name[0].split()),' '.join(name[1].split())]
        return ' '.join(name) + ' | ' + self.perform_camel_case_splitting(name[0]).lower() + '\n'

    def read_in_chunks(self, file_path, n):
        with open(file_path) as fh:
            while True:
                lines = list(islice(fh, n))
                if lines: yield lines
                else:     break

    # Use a specialized POS tagger for tagging words in method names (Gupta POS tagging of method names [21] in MSR paper)
    # Loop over each word tagged as verb from the POS tagger to filter out using following rules:
    #     . Method names starting with third person singular verbs are ignored (VP::3PS)
    #     . If the verb is in past or past participle tense then it is discarded (VP::pp,VP::pastV)
    #     . Linking verbs like is am are has can are discarded (VP::irV)
    #     . Gerunds followed by a past tense verb are ignored (VP::ingV)
    def get_main_verb_from_method_name(self, verbs_names):

        with open('POSSE/Input/temp_methods_1.input','w') as f:
            for v_n in verbs_names:
                name = v_n[0][1]
                f.write(self.prepare_method_input_line(name))

        os.system('cd POSSE/Scripts && ./mainParserChunk.pl ../Input/temp_methods_1.input \"M\"')

        verbs = []
        for lines in self.read_in_chunks('POSSE/Output/temp_methods_1.input.chunked', 5):
            line1 = lines[1][2:].strip()
            line2 = lines[2][2:].strip()
            line0 = lines[0][5:].strip()

            #print (line1)
            phrases = []
            poss = []
            for temp1 in [l1.strip().split(' ') for l1 in line1.split(')')]:
                if len(temp1) > 1:
                    temp1 = [t.replace('(','') for t in temp1]
                    temp1 = [t.replace(',','') for t in temp1]
                    temp1 = [(temp1[0], x) for x in temp1[1:]]
                    poss.append(temp1)

            #print (line2)
            for temp2 in [l2.strip() for l2 in line2.split(' ')]:
                temp2 = temp2.split(':')
                if len(temp2) > 1:
                    if temp2[0][0] == '[':
                        temp2[0] = temp2[0].replace('[','')
                        temp2[0] = temp2[0].replace(']','')
                        phrases.append(temp2)

            c = 0
            appended = False
            if len(phrases) == len(poss):
                for phr, pos in zip(phrases,poss):

                    if c == 0:
                        if phr[1] == 'VP':
                            if any([p[1]=='3PS' for p in pos]):
                                verbs.append('')
                                appended = True
                                continue

                    if phr[1] == 'VP':
                        for p in pos:
                            if p[1] != 'baseV':
                                continue

                            if phr[0] == p[0]:
                                verbs.append([phr[0], line0])
                                appended = True
                                continue

                if appended == False:
                    verbs.append('')
                c += 1
            else:
                verbs.append('')

        return verbs

    def find_pairs(self):

        reader = Reader()
        reader.data_path = self.data_path
        extension = "java"
        reader.pattern = "*." + extension
        reader.get_file_list()
        print("\n")
        print("Got list of files from the data directory")
        print("To view these filenames open etc/filenames.txt")

        files = reader.files
        counter_files = 0
        print ("\n")
        print ("Extracting method names with leading comments")
        comment_name_pair = []
        for file in files:
            if counter_files % 100 == 0: print(counter_files, '/', len(files))

            self.content = reader.read_file(file)
            extracted_method = regex.findall(self.method_comment_regex, self.content)

            for e in extracted_method:

                comment_name_pair.append((' '.join(e[0].split("/*")[-1].split()), [e[-3],e[-2]]))

            counter_files += 1


        # identifying leading comment
        comment_name_pair_filtered = []
        for cn in comment_name_pair:
            phrases = self.identify_desc_lead_comment(cn[0])
            if not phrases: continue
            for p in phrases: comment_name_pair_filtered.append((p,cn[1]))

        # pos tagging using stanford
        all_phrases = [p[0] for p in comment_name_pair_filtered]
        all_phrases = self.stanf_pos_tagger.tag_sents(all_phrases)
        all_names = [p[1] for p in comment_name_pair_filtered]
        all_phrases_names = [(c,n) for c,n in zip(all_phrases,all_names)]

        # reject phrase which contain past tense verb VBD
        all_phrases_names_filtered = []
        for p in all_phrases_names:
            if any(x[1]=='VBD' for x in p[0]):
                continue
            all_phrases_names_filtered.append(p)

        # get the verbs from the phrases
        verbs_names = []
        for p,n in all_phrases_names_filtered:
            #print ('**************************************')

            temp = []
            temp.append(' '.join([x[0] for x in p]))
            count = 0
            prev_word = ('', '')
            for w in p:

                count += 1

                # leave the word alone if it already is declared a verb
                if w[1].startswith('V'):
                    temp.append((w,n,count,prev_word))
                    continue

                # check wordnet entry for this word
                if len(wn.synsets(w[0],pos=wn.VERB))>0:
                    temp.append(((w[0],'VB'),n, count,prev_word))
                    continue

                prev_word = w


            verbs_names.extend([temp])

        # apply rules to get the main verb
        verb_names_new = self.get_main_verb_from_comment(verbs_names)
        verbs = self.get_main_verb_from_method_name(verb_names_new)

        pairs = {}
        for x,y in zip(verb_names_new, verbs):
            if len(x) < 2 or len(y) < 2:
                continue

            v1 = x[0][0][0]
            v2 = y[0]
            if v1!=v2:
                if (v1, v2) in pairs:
                    pairs[(v1,v2)][0] += 1
                    pairs[(v1,v2)].extend([x[1],y[1]])
                elif (v2, v1) in pairs:
                    pairs[(v2, v1)][0] += 1
                    pairs[(v2, v1)].extend([x[1],y[1]])
                else:
                    pairs[(v1, v2)] = [1, x[1],y[1]]


        if not os.path.exists(self.output_path + self.project_name): os.makedirs(self.output_path + self.project_name)

        #write the pairs onto file
        with open(self.output_path + self.project_name + '.csv', 'w') as f:
            for p, v in pairs.items():
                f.write(','.join(p) + ',' + ','.join([str(e) for e in v]) + '\n')


