import sys, time, json, traceback
from collections import namedtuple
from pathlib import Path
import copy
import csv
from inf_tools import print_log, get_edf_filenames, get_channel_labels
import inf_narco_app as narcoApp


class MissingRequiredChannelError(Exception):
    pass


class RunStagesError(Exception):
    def __init__(self, message, edf_filename=''):
        self.message = message
        if isinstance(edf_filename, Path):
            edf_filename = str(edf_filename)
        self.edf_filename = edf_filename


def run_using_json_file(json_file: str):
    output_subpath = 'stanford_stages'
    if json_file is None:
        print_log('Requires a json_file for input.  Nothing done', 'error')
        return

    if not isinstance(json_file, Path):
        json_file = Path(json_file)

    if not json_file.is_file():
        print_log(f'Could not locate json processing file ({str(json_file)}. Nothing done.', 'error')
        return

    with open(json_file, 'r') as fid:
        json_dict: dict = json.load(fid)

    '''
    json_dict['channel_labels'] = {
        'central3': json_dict.pop('C3'),
        'central4': json_dict.pop('C4'),
        'occipital1': json_dict.pop('O1'),
        'occipital2': json_dict.pop('O2'),
        'eog_l': json_dict.pop('EOG-L'),
        'eog_r': json_dict.pop('EOG-R'),
        'chin_emg': json_dict.pop('EMG')
    }
    '''
    # Need to pop any keys here that are not found as part of inf_config class.
    psg_path = None
    if 'edf_pathname' in json_dict:
        psg_path = json_dict.pop('edf_pathname')
        edf_files = get_edf_filenames(psg_path)
    elif 'edf_filename' in json_dict:
        edf_files = [json_dict.pop('edf_filename')]
    elif 'edf_files' in json_dict:
        edf_files = json_dict.pop('edf_files')
    else:
        print_log(f'No edf file or pathname specified in json file ({str(json_file)}. Nothing done.', 'error')
        return -1

    num_edfs = len(edf_files)
    if num_edfs == 0:
        if 'edf_pathname' is not None:
            print_log(
                f'{num_edfs} .edf files found at the edf_pathname ("{psg_path}") specified in "{json_file}"!  '
                f'Nothing to do.', 'error')
        else:
            print_log(f'{num_edfs} .edf files listed in json file ({json_file})!  Nothing to do.', 'error')
        return 0

    if 'output_path' not in json_dict or len(json_dict['output_path'].strip()) == 0:
        if 'output_path' in json_dict:
            del (json_dict['output_path'])
        # only need this psg_path to be guaranteed in this instance, where there is no output path
        # specified.  In this case, we don't know if we have a psg_path for sure, but we are
        # guaranteed to have edf_files in a list.
        psg_path = str(Path(edf_files[0]).parent)
        output_path = Path(psg_path) / output_subpath
        print_log('No output path speficied.  Setting path to: ' + str(output_path), 'warning')
    else:
        output_path = Path(json_dict.pop('output_path'))

    if not output_path.is_dir():
        output_path.mkdir(parents=True)
        if not output_path.is_dir():
            print_log('Could not find or create output directory (' + str(output_path) + '!  QUITTING', 'error')
            return -1
        else:
            print_log('Created ' + str(output_path), 'debug')
    else:
        print_log('Found ' + str(output_path), 'debug')

    if 'edf_pathname' in json_dict:
        print_log(f'{num_edfs} .edf files found at the edf_pathname ("{psg_path}") specified in "{json_file}".', 'info')
    else:
        print_log(f'{num_edfs} .edf files listed in json file ({json_file}) for processing. '
                  f'Output folder is: {str(output_path)}\n')

    # Put this back into our json configuration ...
    json_dict['output_path'] = str(output_path)

    # Check for .csv file containing lights out/on information
    lights_edf_dict = {}
    if 'lights_off_on_filename' in json_dict:
        lights_filename = json_dict['lights_off_on_filename'].strip()
        if lights_filename != "":
            lights_filename = Path(lights_filename)
            if not lights_filename.exists():
                print_log(f'Could not find the "lights_off_on_filename" specified in the .json configuration file. ({str(lights_filename)}','warning')
            else:
                print_log(f'Loading lights off/on information from "{str(lights_filename)}')
                lights_edf_dict: dict = load_lights_from_csv_file(lights_filename)

    # Previously called run_with_edf_files() here
    start_time = time.time()
    pass_fail_dictionary = dict.fromkeys(edf_files, False)
    # Fail with warning if no .edf files are found ?

    for index, edfFile in enumerate(edf_files):
        try:  # ref: https://docs.python.org/3/tutorial/errors.html
            edf_filename = Path(edfFile).name
            msg = f'{index + 1:03d} / {num_edfs:03d}: {edf_filename}\t'

            print_log(msg, 'STAGES')
            # create a copy to avoid issues of making alteration below, such as channel indices ...
            cur_json_dict = copy.deepcopy(json_dict)
            if edf_filename in lights_edf_dict:
                cur_json_dict["lights_on"] = cur_json_dict[edf_filename].get("lights_on", 0)
                cur_json_dict["lights_off"] = cur_json_dict[edf_filename].get("lights_off", 0)
                print_log("Lights on: {cur_json_dict['lights_on']}, Lights off: {cur_json_dict['lights_off']}")

            score, diagnosis_str = run_edf(edfFile, json_configuration=cur_json_dict)
            pass_fail_dictionary[edfFile] = True

            if diagnosis_str is None:
                result_str = 'Narcoleposy Diagnosis Not Performed'
            else:
                result_str = f'[run_stanford_stages.py] Score: {score:0.4f}.  Diagnosis: {diagnosis_str}'
            print_log(result_str, 'STAGES')

        except KeyboardInterrupt:
            print_log('\nUser cancel ...', 'info')
            exit(0)
        except (OSError, ValueError, AttributeError, TypeError) as err:
            print_log("{0}: {1}".format(type(err).__name__, err), 'error')
            traceback.print_exc()
        except KeyError as err:
            print_log("{0}: {1}".format(type(err).__name__, err), 'error')
        except MissingRequiredChannelError as err:
            print_log("Missing required channel(s):\n{0}".format(err), 'error')
        except (RunStagesError, narcoApp.StanfordStagesError) as err:
            print_log(f'{type(err).__name__}: {err.message}  ({err.edf_filename})', 'error')
        except IndexError as err:
            print_log("{0}: {1}".format(type(err).__name__, err), 'error')
            traceback.print_exc()
            print_log('An IndexError may be raised if a the application was running previously with a subset of all '
                      '16 models and is not running with a greater selection of models. If this is the case, '
                      'delete the cached hypnodensity.pkl file and run the software again in order to create the '
                      'nesseary hypnodensity information for all models being used.')

        except:
            # print("Unexpected error:", sys.exc_info()[0])
            print_log("Unexpected error " + str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1]), 'error')
            traceback.print_exc()

    # So many options in python for this
    num_pass = sum(pass_fail_dictionary.values())
    # numPass = [passFailDictionary.values()].count(True)
    # numPass = len([t for t in passFailDictionary.values() if t])

    num_fail = num_edfs - num_pass

    elapsed_time = time.time() - start_time

    print_log(f'{num_edfs} edf files processed in {elapsed_time / 60:0.1f} minutes.\n'
              f'\t{num_pass} processed successfully\n\t{num_fail} had errors')

    if num_fail > 0:
        fail_index = 1
        print_log('The following file(s) failed:', 'warning')
        for (filename, passed) in pass_fail_dictionary.items():
            if not passed:
                print_log(filename)
                fail_index = fail_index + 1


def run_edf(edf_file, json_configuration: {}):
    if not isinstance(edf_file, type(Path)):
        edf_file = Path(edf_file)
    if not edf_file.is_file():
        err_msg = 'edf_file is not a file'
        # print(err_msg)
        raise RunStagesError(err_msg, edf_file)

    print_log("Processing {filename:s}".format(filename=str(edf_file)))
    # display_set_selection(edf_channel_labels_found)

    # Build up our dictionary / channel index mapping
    if 'channel_indices' not in json_configuration:
        label_dictionary = json_configuration.get("channel_labels", None)
        if label_dictionary is None:
            err_msg = 'Either "channel_indices" or "channel_labels" key is required in the json configuration.  ' \
                      'Neither was found.'
            raise RunStagesError(err_msg, edf_file)

        edf_channel_indices_available = dict()
        edf_channel_labels_found = get_channel_labels(edf_file)

        for generic_label, edf_label in label_dictionary.items():
            if isinstance(edf_label, list):
                edf_label_set = set(edf_label)
                edf_label_set = edf_label_set.intersection(edf_channel_labels_found)
                if len(edf_label_set) > 0:
                    edf_channel_indices_available[generic_label] = edf_channel_labels_found.index(edf_label_set.pop())
                else:
                    print_log('{0:s} not found'.format(generic_label), 'debug')
            else:
                if edf_label.lower().strip() == 'none' or edf_label.strip() == '':
                    continue
                elif edf_label in edf_channel_labels_found:
                    edf_channel_indices_available[generic_label] = edf_channel_labels_found.index(edf_label)
                else:
                    print_log('{0:s} not found'.format(edf_label), 'debug')

        # Now we have prepped everything, so let's see if we actually have what we need or not.
        can_continue, cannot_continue_msg = True, ''
        if 'central3' not in edf_channel_indices_available and 'central4' not in edf_channel_indices_available:
            cannot_continue_msg += 'Required central EEG channel missing.\n'
            can_continue = False
        if not ('occipital1' in edf_channel_indices_available or 'occipital2' in edf_channel_indices_available):
            cannot_continue_msg += 'Required occipital EEG channel missing.\n'
            can_continue = False
        if 'eog_l' not in edf_channel_indices_available:
            cannot_continue_msg += 'Required L-EOG channel is missing.\n'
            can_continue = False
        if 'eog_r' not in edf_channel_indices_available:
            cannot_continue_msg += 'Required R-EOG channel is missing.\n'
            can_continue = False
        if 'chin_emg' not in edf_channel_indices_available:
            cannot_continue_msg += 'Required chin EMG channel is missing.\n'
            can_continue = False
        if not can_continue:
            print_log(cannot_continue_msg, 'debug')
            raise MissingRequiredChannelError(cannot_continue_msg)
        json_configuration["channel_indices"] = edf_channel_indices_available

    return narcoApp.main(str(edf_file), json_configuration)


def load_lights_from_csv_file(lights_filename):
    lights_dict: dict = {}
    if not Path(lights_filename).exists():
        print_log("Lights filename does not exist: {lights_filename}", 'warning')
    else:
        with open(lights_filename) as fid:
            f_csv = csv.DictReader(fid)
            headings = next(f_csv)
            Row = namedtuple('Row', ['filename', 'lights_on', 'lights_off'])
            for line in f_csv:
                row = Row(*line)
                lights_dict[line[0]] = dict(zip(('lights_on', 'lights_off'), line[1:2]))
                lights_dict[row.filename] = {'lights_on': row.lights_on, 'lights_off': row.lights_off}

    return lights_dict


def print_usage(tool_name='run_stanford_stages.py'):
    print("Usage:\n\t", tool_name, " <json config file>")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print_usage(args[0])
    else:
        run_using_json_file(json_file=args[1])
