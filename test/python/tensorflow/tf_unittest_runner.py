# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import unittest
import sys
import argparse
import os
import re
import fnmatch
import time
from datetime import timedelta
import warnings

this_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_script_dir + '/../../../tools')
from test_utils import PyTestManifestParser

import multiprocessing
mpmanager = multiprocessing.Manager()
mpmanager_return_dict = mpmanager.dict()

try:
    import xmlrunner
except:
    os.system('pip install unittest-xml-reporting')
    import xmlrunner
os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
"""
tf_unittest_runner is primarily used to run tensorflow python 
unit tests using ngraph
"""


def main():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '--tensorflow_path',
        help=
        "Specify the path to Tensorflow source code. Eg:ngraph-bridge/build_cmake/tensorflow \n",
        required=True)
    optional.add_argument(
        '--list_tests',
        help=
        "Prints the list of test cases in this package. Eg:math_ops_test.* \n")
    optional.add_argument(
        '--list_tests_from_file',
        help=
        """Reads the test names/patterns specified in a manifest file and displays a consolidated list. 
        Eg:--list_tests_from_file=tests_linux_ie_cpu.txt""")
    optional.add_argument(
        '--run_test',
        help=
        "Runs the testcase(s), specified by name or pattern. Eg: math_ops_test.DivNoNanTest.testBasic or math_ops_test.*"
    )
    optional.add_argument(
        '--run_tests_from_file',
        help="""Reads the test names specified in a manifest file and runs them. 
        Eg:--run_tests_from_file=tests_to_run.txt""")
    optional.add_argument(
        '--xml_report',
        help=
        "Generates results in xml file for jenkins to populate in the test result \n"
    )
    optional.add_argument(
        '--verbose',
        action="store_true",
        help="Prints standard out if specified \n")
    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    xml_report = arguments.xml_report

    if (arguments.list_tests):
        test_list = PyTestManifestParser().get_test_list(arguments.tensorflow_path,
                                  arguments.list_tests)
        print('\n'.join(test_list[0]))
        print('Total:', len(test_list[0]))
        return None, None

    if (arguments.list_tests_from_file):
        test_list, skip_list = PyTestManifestParser().read_tests_from_manifest(
            arguments.list_tests_from_file, arguments.tensorflow_path)
        print('\n'.join(test_list))
        print('Total:', len(test_list), 'Skipped:', len(skip_list))
        return None, None

    if (arguments.run_test):
        invalid_list = []
        start = time.time()
        test_list = PyTestManifestParser().get_test_list(arguments.tensorflow_path, arguments.run_test)
        for test in test_list[1]:
            if test is not None:
                invalid_list.append(test_list[1])
                result_str = "\033[91m INVALID \033[0m " + test + \
                '\033[91m' + '\033[0m'
                print('TEST:', result_str)
        test_results = run_test(test_list[0], xml_report,
                                (2 if arguments.verbose else 0))
        elapsed = time.time() - start
        print("\n\nTesting results\nTime elapsed: ",
              str(timedelta(seconds=elapsed)))
        return check_and_print_summary(test_results, test_list[1])

    if (arguments.run_tests_from_file):
        all_test_list = []
        invalid_list = []
        start = time.time()
        list_of_tests = PyTestManifestParser().read_tests_from_manifest(arguments.run_tests_from_file,
                                                 arguments.tensorflow_path)[0]
        test_results = run_test(list_of_tests, xml_report,
                                (2 if arguments.verbose else 0))
        elapsed = time.time() - start
        print("\n\nTesting results\nTime elapsed: ",
              str(timedelta(seconds=elapsed)))
        return check_and_print_summary(test_results, invalid_list)




def func_utrunner_testcase_run(return_dict, runner, aTest):
    # This func runs in a separate process
    try:
        test_result = runner.run(aTest)
        return_dict[aTest.id()] = {
            'wasSuccessful':
            test_result.wasSuccessful(),
            'failures':
            test_result.failures,
            'errors':
            test_result.errors,
            'skipped': [('', test_result.skipped[0][1])] if
            (test_result.skipped) else None
        }
    except Exception as e:
        #print('DBG: func_utrunner_testcase_run test_result.errors', test_result.errors, '\n')
        return_dict[aTest.id()] = {
            'wasSuccessful': False,
            'failures': [('', test_result.errors[0][1])],
            'errors': [('', test_result.errors[0][1])],
            'skipped': []
        }


def run_singletest_in_new_child_process(runner, aTest):
    mpmanager_return_dict.clear()
    return_dict = mpmanager_return_dict
    p = multiprocessing.Process(
        target=func_utrunner_testcase_run, args=(return_dict, runner, aTest))
    p.start()
    p.join()

    #  A negative exitcode -N indicates that the child was terminated by signal N.
    if p.exitcode != 0:
        error_msg = '!!! RUNTIME ERROR !!! Test ' + aTest.id(
        ) + ' exited with code: ' + str(p.exitcode)
        print(error_msg)
        return_dict[aTest.id()] = {
            'wasSuccessful': False,
            'failures': [('', error_msg)],
            'errors': [('', error_msg)],
            'skipped': []
        }
        return return_dict[aTest.id()]

    test_result_map = return_dict[aTest.id()]
    return test_result_map


def run_test(test_list, xml_report, verbosity=0):
    """
    Runs a specific test suite or test case given with the fully qualified 
    test name and prints stdout.

    Args:
    test_list: This is the list of tests to run,filtered based on the 
    regex_input passed as an argument.
    Example: --run_test=math_ops_test.A*   
    verbosity: Python verbose logging is set to 2. You get the help string 
    of every test and the result.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    succeeded = []
    failures = []
    skipped = []
    run_test_counter = 0
    if xml_report is not None:
        for testpattern in test_list:
            tests = loader.loadTestsFromName(testpattern)
            suite.addTest(tests)
        with open(xml_report, 'wb') as output:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            test_result = xmlrunner.XMLTestRunner(
                output=output, verbosity=verbosity).run(suite)
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__
            failures.extend(test_result.failures)
            failures.extend(test_result.errors)
            succeeded.extend(test_result.successes)

        summary = {"TOTAL": test_list, "PASSED": succeeded, "FAILED": failures}
        return summary
    else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        for testpattern in test_list:
            testsuite = loader.loadTestsFromName(testpattern)
            for aTest in testsuite:
                print()
                run_test_counter += 1
                print('>> >> >> >> ({}) Testing: {} ...'.format(
                    run_test_counter, aTest.id()))
                start = time.time()
                test_result_map = run_singletest_in_new_child_process(
                    runner, aTest)
                elapsed = time.time() - start
                elapsed = str(timedelta(seconds=elapsed))

                if test_result_map['wasSuccessful'] == True:
                    succeeded.append(aTest.id())
                    result_str = " \033[92m OK \033[0m " + aTest.id()
                elif 'failures' in test_result_map and bool(
                        test_result_map['failures']):
                    failures.append(test_result_map['failures'])
                    result_str = " \033[91m FAIL \033[0m " + aTest.id() + \
                        '\n\033[91m' + ''.join(test_result_map['failures'][0][1]) + '\033[0m'
                elif 'errors' in test_result_map and bool(
                        test_result_map['errors']):
                    failures.append(test_result_map['errors'])
                    result_str = " \033[91m FAIL \033[0m " + aTest.id() + \
                        '\n\033[91m' + ''.join(test_result_map['errors'][0][1]) + '\033[0m'

                if 'skipped' in test_result_map and bool(
                        test_result_map['skipped']):
                    skipped.append(test_result_map['skipped'])
                print('took', elapsed, 'RESULT =>', result_str)
        summary = {
            "TOTAL": test_list,
            "PASSED": succeeded,
            "SKIPPED": skipped,
            "FAILED": failures,
        }
        return summary


def check_and_print_summary(test_results, invalid_list):
    print('========================================================')
    print("TOTAL: ", len(test_results['TOTAL']))
    print("PASSED: ", len(test_results['PASSED']))
    if len(test_results['SKIPPED']) > 0:
        print("   with skipped: ", len(test_results['SKIPPED']))
    print("FAILED: ", len(test_results['FAILED']))

    if (len(invalid_list) > 0):
        print("INVALID: ", len(invalid_list))

    print('========================================================\n')

    if len(test_results['FAILED']) == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = main()
        if status == False:
            raise Exception("Tests failed")
