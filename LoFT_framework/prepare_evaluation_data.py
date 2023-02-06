import json
from collections import Counter
import random, argparse
random.seed(233)

def read_verifier_output(verifier_output_file):
    examples = json.load(open(verifier_output_file, "r"))

    collected_dict = {}
    for example in examples:
        if example["verification_label"] == "refuted":
            continue
        csv_id = example["csv_id"]
        col_ids = str(example["col_ids"])
        generated_statement = example["statement"]

        if csv_id not in collected_dict:
            collected_dict[csv_id] = {}
        if col_ids not in collected_dict[csv_id]:
            collected_dict[csv_id][col_ids] = []
        
        collected_dict[csv_id][col_ids].append((example["verification_score"], generated_statement))
    
    return collected_dict


def sample_output_for_logicnlg_evaluation(logicnlg_test_file, output_candidate_dict, evaluation_file):
    examples = json.load(open(logicnlg_test_file, "r"))
    output_data = {}

    for csv_file in examples:
        output_data[csv_file] = []
        csv_id = csv_file.split(".")[0]
        col_ids_list = []
        for example in examples[csv_file]:
            col_ids_list.append(str(example[1]))
        col_ids_count = Counter(col_ids_list)
        col_ids_statement_dict = {}
        for col_ids, count in col_ids_count.items():
            sampled_list = sorted(output_candidate_dict[csv_id][col_ids], key=lambda x: x[0], reverse=True)
            
            threshold_idx = 0
            for idx, i in enumerate(sampled_list):
                if i[0] < 0.7:
                    threshold_idx = idx

            sampled_list = sampled_list[:max(count, threshold_idx)]
            col_ids_statement_dict[col_ids] = random.sample(sampled_list, count)
        
        for col_ids in col_ids_list:
            output_data[csv_file].append(col_ids_statement_dict[col_ids].pop()[1])
    
    json.dump(output_data, open(evaluation_file, "w"), indent=4)
    print("Evaluation file saved to {}".format(evaluation_file))
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_output_file", type=str, default="LoFT_predict/inference_with_verification_result.json")
    parser.add_argument("--logicnlg_test_file", default = "../LoFT_data_processing/data/logicnlg/test_lm.json", type=str)

    output_candidate_dict = read_verifier_output(verifier_output_file)
    evaluation_file = "LoFT_predict/evaluation.json"
    output_data = sample_output_for_logicnlg_evaluation(logicnlg_test_file, output_candidate_dict, evaluation_file)
    
main()