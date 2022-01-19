import sys, time, json, traceback

# from collections import namedtuple
from pathlib import Path
import copy
import csv
import inf_tools
from inf_tools import print_log
import inf_narco_app as narco_app


class MissingRequiredChannelError(Exception):
    pass


class ConfigurationStagesError(Exception):
    pass


class RunStagesError(Exception):
    def __init__(self, message, edf_filename=''):
        self.message = message
        if isinstance(edf_filename, Path):
            edf_filename = str(edf_filename)
        self.edf_filename = edf_filename


def run_using_json_file(json_file: str, progress_cb=None):
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

    # bypass looking for edf data in the case when channel indices are not provided, the channel labels are provided and
    # they are explicitly set to 'None'.
    bypass_edf_check = json_dict.get('bypass_edf_check', False) or 'channel_indices' not in json_dict and 'channel_labels' in json_dict and all(
        [label.lower() == 'none' for label in json_dict['channel_labels'].values()])

    if bypass_edf_check:
        # This will let us bypass the channel label configuration later without raising an exception.
        print("Bypassing edf check")
        json_dict['channel_indices'] = []
        json_dict['bypass_edf_check'] = bypass_edf_check

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
    edf_files = []
    if 'edf_pathname' in json_dict:
        psg_path = json_dict.pop('edf_pathname')
        if not bypass_edf_check:
            edf_files = inf_tools.get_edf_filenames(psg_path)
        else:
            unwanted_edf_files = inf_tools.get_edf_files(psg_path)

            # b = inf_tools.get_files_with_ext('F:\\jazz\\testing', 'h5')
            h5_files = inf_tools.get_h5_files(psg_path)

            if len(h5_files):
                # determine basename of these files... note that we currently save .features and .hypnodensity files with .h5 suffx
                # as well as encoding files
                prefix_names = [b.stem.partition('.hypnodensity')[0].partition('.features')[0] for b in
                                h5_files]  # file1.hypnodensity.h5 and file1.h5 --> file1 and file1
                # remove any duplicates that are found
                prefix_names = list(set(prefix_names))
                # then create mock-up .edf files with the remaining basenames, provided that they are not in the list of .edf files already.
                edf_names = [b.stem.lower() for b in unwanted_edf_files]
                p = Path(psg_path)
                edf_files = []
                for name in prefix_names:
                    if name.lower() not in edf_names:
                        edf_files.append(str(p / (name + '.edf')))
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

    # Check for .evt file(s) containing start/stop events to exclude from the analysis (e.g. bad data)
    data_exclusion_key = 'exclusion_events_pathname'
    data_exclusion_path = json_dict.get(data_exclusion_key, None)
    if data_exclusion_path is not None:
        data_exclusion_path = Path(data_exclusion_path)
        if not data_exclusion_path.is_dir():
            err_msg = f'A {data_exclusion_key} entry was found in the json file, but the path ("{str(data_exclusion_path)}") could not be found.  Correct the pathname or remove it.'
            print_log(err_msg, 'error')
            # data_exclusion_path = None
            raise ConfigurationStagesError(err_msg)
        else:
            print_log(f'Using "{str(data_exclusion_path)}" as path containing data exclusion event file(s).')

    # Check for .csv file containing lights out/on information
    lights_filename_key = 'lights_filename'
    lights_edf_dict = {}
    if lights_filename_key in json_dict:
        lights_filename = json_dict[lights_filename_key].strip()
        if lights_filename != "":
            lights_filename = Path(lights_filename)
            if not lights_filename.exists():
                err_msg = f'Could not find the "{lights_filename_key}" key specified in the .json configuration file.  Correct the filename or remove it: {str(lights_filename)}'
                print_log(err_msg, 'error')
                raise ConfigurationStagesError(err_msg)
            else:
                print_log(f'Loading lights off/on information from "{str(lights_filename)}')
                lights_edf_dict: dict = load_lights_from_csv_file(lights_filename)

    # Previously called run_with_edf_files() here
    start_time = time.time()
    pass_fail_dictionary = dict.fromkeys(edf_files, False)
    # Fail with warning if no .edf files are found ?

    # Lights on/off order of preference is
    # 1.  If there is a lights on/off file with an entry for the current .edf, its value is used
    # 2.  If that is missing, then the lights_off and lights_on keys will be used if they are listed in the .json configuration file
    # 3.  If this is missing, the default value will be None for each lights_off and lights_on field.
    #     A value of None is handled as no entry given which and the entire study will be used
    #     (i.e. lights out assumed to coincides with PSG start and lights on coincides with the end of the recording).

    default_lights_off = json_dict.get("inf_config", {}).get("lights_off", None)
    default_lights_on = json_dict.get("inf_config", {}).get("lights_on", None)

    edf_files = sorted(edf_files)
    last_update_time = time.time()
    for index, edfFile in enumerate(edf_files):
        try:  # ref: https://docs.python.org/3/tutorial/errors.html
            edf_filename = Path(edfFile).name

            msg = f'{index + 1:03d} / {num_edfs:03d}: {edf_filename}'
            print_log(msg, 'STAGES')

            # create a copy to avoid issues of making alteration below, such as channel indices ...
            cur_json_dict = copy.deepcopy(json_dict)
            # Give some flexibility to whether the .edf file name is given or just the basename (sans extension)
            file_key = None
            if edf_filename in lights_edf_dict:
                file_key = edf_filename
            elif edf_filename.partition('.')[0] in lights_edf_dict:
                file_key = edf_filename.partition('.')[0]
            if file_key is not None:
                cur_json_dict["inf_config"]["lights_off"] = lights_edf_dict[file_key].get("lights_off",
                                                                                          default_lights_off)
                cur_json_dict["inf_config"]["lights_on"] = lights_edf_dict[file_key].get("lights_on", default_lights_on)
                print_log(f"Lights off: {cur_json_dict['inf_config']['lights_off']}, Lights on: "
                          f"{cur_json_dict['inf_config']['lights_on']}")

            if data_exclusion_path is not None:
                #  data_exclusion_filename = str(data_exclusion_path / (edf_filename.partition('.')[0] + '.evt'))
                # cur_json_dict['bad_data_events'] = get_bad_data_events(data_exclusion_filename)
                data_exclusion_file = data_exclusion_path / (edf_filename.partition('.')[0] + '.evt')
                if data_exclusion_file.is_file():
                    log_msg = f'Data exclusion file found: {str(data_exclusion_file)}'
                    cur_json_dict["inf_config"]["bad_data_filename"] = str(data_exclusion_file)
                else:
                    log_msg = f'No data exclusion file found for current study ({edf_filename}): {str(data_exclusion_file)}'
                print_log(log_msg, 'info')

            score, diagnosis_str = run_study(edfFile, json_configuration=cur_json_dict,
                                             bypass_edf_check=bypass_edf_check)
            pass_fail_dictionary[edfFile] = True

            if diagnosis_str is None:
                result_str = 'Narcoleposy Diagnosis Not Performed'
            else:
                result_str = f'[run_stanford_stages.py] Score: {score:0.4f}.  Diagnosis: {diagnosis_str}'

            if progress_cb is not None:
                time_elapsed_from_last_update = time.time() - last_update_time
                if time_elapsed_from_last_update > 3:
                    last_update_time = time.time()
                    progress_cb(index/num_edfs*100, index, edfFile, msg+' '+result_str)

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
        except (RunStagesError, narco_app.StanfordStagesError) as err:
            print_log(f'{type(err).__name__}: {err.message}  ({err.edf_filename if err.edf_filename != "" else edf_filename})', 'error')
        except IndexError as err:
            print_log("{0}: {1}".format(type(err).__name__, err), 'error')
            traceback.print_exc()
            print_log('An IndexError may be raised if the application was previously run with a subset of all '
                      '16 models and is now using a greater or different selection of models. Try deleting the '
                      'cached hypnodensity.(pkl/h5) file(s) and run the software again to generate the '
                      'necessary hypnodensity information for the current configuration.')
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


def run_study(edf_file, json_configuration: {}, bypass_edf_check: bool = False):
    if not isinstance(edf_file, type(Path)):
        edf_file = Path(edf_file)
    if not bypass_edf_check and not edf_file.is_file():
        err_msg = 'edf_file is not a file'
        raise RunStagesError(err_msg, edf_file)

    print_log("Processing {filename:s}".format(filename=str(edf_file)))
    # display_set_selection(edf_channel_labels_found)

    if 'inf_config' in json_configuration:
        json_configuration['inf_config']['lights_off'] = edftime2elapsedseconds(edf_file,
                                                                                json_configuration['inf_config'].get(
                                                                                    'lights_off', None))
        json_configuration['inf_config']['lights_on'] = edftime2elapsedseconds(edf_file,
                                                                               json_configuration['inf_config'].get(
                                                                                   'lights_on', None))

    # Build up our dictionary / channel index mapping
    if 'channel_indices' not in json_configuration:
        label_dictionary = json_configuration.get("channel_labels", None)
        if label_dictionary is None:
            err_msg = 'Either "channel_indices" or "channel_labels" key is required in the json configuration.  ' \
                      'Neither was found.'
            raise RunStagesError(err_msg, edf_file)

        edf_channel_indices_available = dict()
        edf_channel_labels_found = inf_tools.get_channel_labels(edf_file)

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

    return narco_app.main(str(edf_file), json_configuration)


def get_bad_data_events(events_filename):
    events_dict = {}
    if not Path(events_filename).exists():
        print_log(f"File containing events to exclude not found: {events_filename}", 'warning')
    else:
        with open(events_filename) as fid:
            f_csv = csv.reader(fid)
            for line in f_csv:
                start = line[0]
                stop = line[1]
                if start not in events_dict or float(stop) > float(events_dict[start]):
                    events_dict[start] = stop
    return events_dict


def load_lights_from_csv_file(lights_filename):
    lights_dict: dict = {}
    if not Path(lights_filename).exists():
        print_log(f"Lights filename does not exist: {lights_filename}", 'warning')
    else:
        with open(lights_filename) as fid:
            # f_csv = csv.DictReader(fid)
            f_csv = csv.reader(fid)
            # headings = next(f_csv)
            # Row = namedtuple('Row', ['filename', 'lights_on', 'lights_off'])
            for line in f_csv:
                # row = Row(*line)
                lights_dict[line[0]] = dict(zip(('lights_off', 'lights_on'), line[1:3]))
                # lights_dict[row.filename] = {'lights_on': row.lights_on, 'lights_off': row.lights_off}

    return lights_dict


def edftime2elapsedseconds(edf_file, time_value):
    if isinstance(time_value, str) and ":" in time_value:
        if edf_file is None or not Path(edf_file).exists():
            raise (ValueError(
                'Cannot convert time stamp to elapsed seconds from the study start because an EDF file, which contains the study start time, was not found.'))
        else:
            study_start_time_seconds = inf_tools.get_study_starttime_as_seconds(edf_file)
            if study_start_time_seconds is None:
                raise (RunStagesError('Unable to find start time for edf file'))
            time_hh_mm_ss = time_value.split(':')
            convert_hh_mm_ss = [3600, 60, 1, 0.001]
            time_value_seconds = 0
            for idx, value in enumerate(time_hh_mm_ss):
                time_value_seconds = time_value_seconds + int(value) * convert_hh_mm_ss[idx]

            elapsed_seconds = time_value_seconds - study_start_time_seconds
            if elapsed_seconds < 0:
                elapsed_seconds = elapsed_seconds + 24 * 3600
    else:
        elapsed_seconds = time_value

    return elapsed_seconds


def print_usage(tool_name='run_stanford_stages.py'):
    print("Usage:\n\t", tool_name, " <json config file>")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print_usage(args[0])
    else:
        run_using_json_file(json_file=args[1])
