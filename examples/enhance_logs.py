# we need to preprocess the results from log files
import json
import re
import sys
from pathlib import Path
from iohblade.loggers import ExperimentLogger
from llamea.ast_features import extract_ast_features

PATTERN = re.compile(
    r"analysis,\s*try to\s+"
    r"(increase|decrease)\s+the\s+(.+?)\s+of the solution\.?",
    re.IGNORECASE,
)

def extract_archive_instruction(text: str):
    """
    Look for 'Based on archive analysis, try to <increase/decrease> the <feature> of the solution.'
    Return (direction, feature) or ("", "") if not found.
    """
    match = PATTERN.search(text)
    if not match:
        return "", ""
    direction = match.group(1).lower().strip()
    feature = match.group(2).strip()
    return direction, feature

def transform_log(log_path="log.jsonl", output_path="log.jsonl"):
    log_path = Path(log_path)
    output_path = Path(output_path)
    with log_path.open("r", encoding="utf-8") as f_log:
        log_lines = f_log.readlines()
    
    with output_path.open("w", encoding="utf-8") as f_out:
        for log_line in log_lines:
            log_obj = json.loads(log_line)
            metadata = log_obj.get("metadata", {})
            try:
                #get ast features from metadata
                ast_features = metadata.get("ast_features", {})
                log_obj["ast_features"] = ast_features
            except Exception as e:
                print(f"Error extracting AST features: {e}")
                log_obj["ast_features"] = {}

            log_obj["archive_direction"] = metadata.get("feature_guidance_action", "")
            log_obj["archive_feature"] = metadata.get("feature_guidance_feature_name", "")
            
            f_out.write(json.dumps(log_obj) + "\n")
    
def parse_operator(log_folder="log"):
    log_file = Path(log_folder) / "log.jsonl"
    conversation_file = Path(log_folder) / "conversationlog.jsonl"
    with log_file.open("r", encoding="utf-8") as f_log:
        log_lines = f_log.readlines()
    with conversation_file.open("r", encoding="utf-8") as f_conv:
        conv_lines = f_conv.readlines()
    # Sanity check: conversationlog should have 2x lines of log.jsonl
    if len(conv_lines) != 2 * len(log_lines):
        print(
            f"Warning: expected {2 * len(log_lines)} lines in conversationlog, "
            f"but found {len(conv_lines)}.",
            file=sys.stderr,
        )

    with log_file.open("w", encoding="utf-8") as f_out:
        for i, log_line in enumerate(log_lines):
            log_obj = json.loads(log_line)
            # Instruction is on every odd line index in terms of objects:
            # line 0 (client), line 1 (gpt), line 2 (client), line 3 (gpt), ...
            instr_idx = 2 * i
            try:
                instr_obj = json.loads(conv_lines[instr_idx])
                instr_text = instr_obj.get("content", "")
            except (IndexError, json.JSONDecodeError):
                instr_text = ""
            if "Refine the strategy of the selected solution to improve it." in instr_text:
                log_obj["operator"] = "refine"
            elif "Generate a new algorithm that is different from the algorithms you have tried before." in instr_text:
                log_obj["operator"] = "random"
            # Add / overwrite fields
            f_out.write(json.dumps(log_obj) + "\n")

def enrich_log(conversation_path="conversationlog.jsonl",
         log_path="log.jsonl",
         output_path="log_enriched.jsonl"):
    conversation_path = Path(conversation_path)
    log_path = Path(log_path)
    output_path = Path(output_path)

    with conversation_path.open("r", encoding="utf-8") as f_conv:
        conv_lines = f_conv.readlines()

    with log_path.open("r", encoding="utf-8") as f_log:
        log_lines = f_log.readlines()

    # Sanity check: conversationlog should have 2x lines of log.jsonl
    if len(conv_lines) != 2 * len(log_lines):
        print(
            f"Warning: expected {2 * len(log_lines)} lines in conversationlog, "
            f"but found {len(conv_lines)}.",
            file=sys.stderr,
        )

    with output_path.open("w", encoding="utf-8") as f_out:
        for i, log_line in enumerate(log_lines):
            log_obj = json.loads(log_line)

            # only process if "ast_features" is not already present
            if "ast_features" in log_obj:
                f_out.write(json.dumps(log_obj) + "\n")
                continue

            code = log_obj["code"] if "code" in log_obj else ""
            if not code.strip():
                print("No code found!", log_obj)
                continue
            
            try:
                ast_features = extract_ast_features(code)
                log_obj["ast_features"] = ast_features
            except Exception as e:
                print(f"Error extracting AST features: {e}")
                log_obj["ast_features"] = {}

            # Instruction is on every odd line index in terms of objects:
            # line 0 (client), line 1 (gpt), line 2 (client), line 3 (gpt), ...
            instr_idx = 2 * i
            try:
                instr_obj = json.loads(conv_lines[instr_idx])
                instr_text = instr_obj.get("content", "")
            except (IndexError, json.JSONDecodeError):
                instr_text = ""

            direction, feature = extract_archive_instruction(instr_text)

            

            # Add / overwrite fields
            log_obj["archive_direction"] = direction
            log_obj["archive_feature"] = feature

            f_out.write(json.dumps(log_obj) + "\n")

    print(f"Enriched log written to: {output_path}")


    # Optional: allow CLI overrides
    # Usage: python enrich_logs.py conversationlog.jsonl log.jsonl log_enriched.jsonl

if __name__ == "__main__":
    for log_folder in ['results/BBOB_guided','results/BBOB_guided2','results/BBOB_guided3', 'results/MABBOB_guided']:
        logger = ExperimentLogger(log_folder, True)
        log_data = logger.get_data()
        
        for index, entry in log_data.iterrows():
            transform_log(log_folder + '/' + entry['log_dir'] + '/log.jsonl', log_folder + '/' + entry['log_dir'] + '/log.jsonl')
            parse_operator(log_folder + '/' + entry['log_dir'])
    
    if False:
        log_folder = '/home/neocortex/repos/BLADE/results/BBOB_guided2/'

        logger = ExperimentLogger('/home/neocortex/repos/BLADE/results/BBOB_guided2', True)
        log_data = logger.get_data()
        
        for index, entry in log_data.iterrows():
            enrich_log(log_folder + entry['log_dir'] + '/conversationlog.jsonl',
                log_folder + entry['log_dir'] + '/log.jsonl',
                log_folder + entry['log_dir'] + '/log2.jsonl')
            # now rename log2.jsonl to log.jsonl and log.jsonl to log_old.jsonl
            Path(log_folder + entry['log_dir'] + '/log.jsonl').rename(log_folder + entry['log_dir'] + '/log_old.jsonl')
            Path(log_folder + entry['log_dir'] + '/log2.jsonl').rename(log_folder + entry['log_dir'] + '/log.jsonl')

    