import os
import json

datasets_config = {
    # "Agriculture": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]},
    # "Climate": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]},
    "Energy": {"seq_len": 36, "prediction_lengths": [12, 24, 36, 48]},
    # "Environment": {"seq_len": 96, "prediction_lengths": [48, 96]},
    # "Health": {"seq_len": 36, "prediction_lengths": [12, 24, 36, 48]},
    # "Security": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]},
    "SocialGood": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]},
    # "Traffic": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]},
    # "new_metr": {"seq_len": 96, "prediction_lengths": [48, 96]},
    # "Economy_sort": {"seq_len": 8, "prediction_lengths": [6, 8, 10, 12]}
}
folder_input = "integrated_predictions_stat/"
for folder_name in datasets_config.keys():
    folder_save = "stat_predictions_stat/" + folder_name
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    full_data_path = os.path.join(folder_save, folder_name+"_length_"+str(datasets_config[folder_name]["prediction_lengths"][-1])+"_predictions.json")
    train_json = json.load(open(folder_input + 'stage1_train_optimized/' + folder_name + "_stage1_optimized.json"))
    val_test_json = json.load(open(folder_input + 'stage2_final_predictions/' + folder_name + "_final_predictions.json"))

    full_json = {}
    for i in train_json:
        full_json[i] = {}
        full_json[i]["Prediction"] = train_json[i]["predicted_values"]
    for j in val_test_json:
        full_json[j] = {}
        full_json[j]["Prediction"] = val_test_json[j]["prediction"]

    with open(full_data_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)


