
import os
import fnmatch
import regex

__author__ = ["Shayan Ali Akbar"]
__email__ =  ["sakbar@purdue.edu"]

class Reader:
    '''
	This class is used to read files from the disk
	Also it can list all the files that are there in the data
	'''

    def __init__(self):
        self.data_path = None  # path for the repo
        self.pattern = None  # extension pattern for regex
        self.files = None  # list of all files in the repo
        self.xml_bug_reports_file = None  # xml file path of bug report

    def get_file_list(self):
        '''
		Get a list of all files from the data dir
		ret1: list of filenames
		'''

        filenames = []
        counter_files = 0
        with open("etc/filenames.txt", "w") as f:
            for root, dirs, files in os.walk(self.data_path):
                for basename in files:
                    if fnmatch.fnmatch(basename, self.pattern):
                        filename = os.path.join(root, basename)
                        f.write(str(counter_files) + "," + filename + "\n")
                        counter_files += 1
                        filenames.append(filename)
        self.files = filenames
        return filenames

    def read_file(self, file_path):
        '''
		Read contents of a single file
		arg1: the path to the file
		ret1: content of the file
		'''
	
        try:
            with open(file_path, encoding='iso-8859-15') as f:
                file_content = f.read()
            return file_content
        except:
            return " "

