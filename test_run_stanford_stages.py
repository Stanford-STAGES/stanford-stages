import run_stanford_stages as narco_class_app


if __name__ == '__main__':

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # exit(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    json_filename = 'C:\\Data\\ml\\stanford_stages.json'
    json_filename = 'F:\\wsc_edf\\stanford_stages.json'
    json_filename = 'H:\Data\Converted\Valid\stanford_stages.json'
    json_filename = 'C:\\Users\\hyatt\\GitHub\\stanford-stages\\run_jazz_training_hypnodensity.json'
    json_filename = 'jazz_training_hypnodensity.json'
    narco_class_app.run_using_json_file(json_filename)
