from unittest import TestLoader, TestResult, TextTestRunner
# Temp remove to allow nextflow to run
#from pathlib import Path
from pprint import pprint
import sys, os


def run_tests():
    #sys.path.insert(0, os.path.dirname(__file__))
    test_loader = TestLoader()
    test_result = TestResult()


    # Use resolve() to get an absolute path
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
    test_directory = str(Path(__file__).resolve().parent / '')

    print(test_directory)

    test_suite = test_loader.discover(test_directory, pattern='test_al*.py')

    test_result = TextTestRunner(buffer=True).run(test_suite)
    #print(test_suite)
    #test_suite.run(result=test_result, buffer = True)


    print(test_result)
    # See the docs for details on the TestResult object
    # https://docs.python.org/3/library/unittest.html#unittest.TestResult

    if test_result.wasSuccessful():
        exit(0)
    else:
        # Here you can either print or log your test errors and failures
        # test_result.errors or test_result.failures
        pprint(test_result.errors)
        pprint(test_result.failures)
        exit(-1)

if __name__ == "__main__":
    run_tests()


