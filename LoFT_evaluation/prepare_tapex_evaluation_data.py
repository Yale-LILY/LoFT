import json, os
import argparse

def prepare_tapex_evaluation_data(input_file, raw_test_file, table_path):
    input_data = json.load(open(input_file, "r"))
    raw_data = json.load(open(raw_test_file, "r"))

    output_data = []
    for csv_file in input_data:
        for statement in input_data[csv_file]:
            cur_example = {"statement": statement}
            cur_example["table_text"] = open(os.path.join(table_path, csv_file), "r").read()
            cur_example["label"] = 1
            cur_example["table_title"] = raw_data[csv_file][0][2]
        
            output_data.append(cur_example)

    os.makedirs("tmp", exist_ok=True)
    json.dump(output_data, open("tmp/test.json", "w"), indent=4)
    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify_file", type=str, required=True)
    parser.add_argument("--raw_test_file", default = "LogicNLG/data/test_lm.json", type=str)
    parser.add_argument("--table_path", default = "LogicNLG/data/all_csv", type=str)

    args = parser.parse_args()
    prepare_tapex_evaluation_data(args.verify_file, args.raw_test_file, args.table_path)


