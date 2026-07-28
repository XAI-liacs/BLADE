import os
import json
import gc
import resource

from modcma import c_maes
import numpy as np

# file_path = '/local/bodasap/BLADE-RAG/iohblade/problems/generated_problems/gpt-5-nano-ELA-Basins_Homogeneous.jsonl'
# # load the json as dictionary
# import json
# dicttest = {}
# with open(file_path, 'r') as f:
#     for line in f:
#         dicttest = json.loads(line)
#         break  # Only read the first line

# namespace = {}
# exec(dicttest['code'], namespace)

# landscape = namespace[dicttest['name']](30)

# x_best, f_best, evaluations, optimizer = c_maes.fmin(
#     landscape.f,
#     np.zeros(30),
#     2.5,
#     100000,
# )

# print(f"Best solution: {x_best}, Best function value: {f_best}, Evaluations: {evaluations}")


directory = '/local/bodasap/BLADE-RAG/iohblade/problems/generated_problems/'
memory_ceiling_mb = 5120
json_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
print(f"Found {len(json_files)} JSONL files in the directory.")


def set_memory_ceiling(extra_memory_mb):
    # modcma reserves a large virtual address space when it is imported.
    # Add the ceiling to the process's current virtual size instead of using
    # the ceiling as an absolute value.
    with open('/proc/self/statm', 'r') as f:
        current_pages = int(f.read().split()[0])

    current_virtual_memory = current_pages * os.sysconf('SC_PAGE_SIZE')
    soft_limit = current_virtual_memory + extra_memory_mb * 1024 ** 2
    previous_limits = resource.getrlimit(resource.RLIMIT_AS)
    _, hard_limit = previous_limits

    if hard_limit != resource.RLIM_INFINITY:
        soft_limit = min(soft_limit, hard_limit)

    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
    return previous_limits


def save_jsonl(file_path, records):
    temp_path = file_path + '.tmp'
    with open(temp_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    os.replace(temp_path, file_path)


print(f"Generated landscapes may use at most {memory_ceiling_mb} MB extra memory.")


for file_number, json_file in enumerate(json_files, start=1):
    print(f"Processing file {file_number}/{len(json_files)}: {json_file}")
    file_path = os.path.join(directory, json_file)

    # Load all lines so each result can be written to its corresponding line.
    with open(file_path, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Resume at file level: skip files whose lines are all complete.
    if all(
        (
            'x_opt_dim_30' in record and 'f_opt_dim_30' in record
        ) or record.get('memory_skipped_dim_30', False)
        for record in records
    ):
        print(f"Skipping completed file: {json_file}")
        continue

    for line_number, dicttest in enumerate(records, start=1):
        # Resume at line level: skip records that already have both results.
        if (
            (
                'x_opt_dim_30' in dicttest
                and 'f_opt_dim_30' in dicttest
            )
            or dicttest.get('memory_skipped_dim_30', False)
        ):
            print(
                f"Skipping completed line {line_number}/{len(records)} "
                f"with ID: {dicttest['id']}"
            )
            continue

        id_alg = dicttest['id']
        name = dicttest['name']
        print(
            f"Processing line {line_number}/{len(records)}: "
            f"{name} with ID: {id_alg}"
        )
        namespace = {}
        landscape = None
        optimizer = None
        previous_memory_limits = set_memory_ceiling(memory_ceiling_mb)
        try:
            exec(dicttest['code'], namespace)
            landscape = namespace[dicttest["name"]](30)

            x_best, f_best, evaluations, optimizer = c_maes.fmin(
                landscape.f,
                np.zeros(30),
                2.5,
                100000,
            )
        except (MemoryError, ValueError) as error:
            # Restore normal process memory before cleanup and checkpointing.
            resource.setrlimit(resource.RLIMIT_AS, previous_memory_limits)

            is_oversized_array = (
                isinstance(error, ValueError)
                and 'array is too big' in str(error)
            )
            if isinstance(error, ValueError) and not is_oversized_array:
                raise

            dicttest['memory_skipped_dim_30'] = True
            namespace.clear()
            del optimizer
            del landscape
            gc.collect()
            save_jsonl(file_path, records)
            print(
                f"Skipping line {line_number} with ID {id_alg}: "
                f"landscape requires too much memory ({error})."
            )
            continue

        # Restore normal process memory before cleanup and checkpointing.
        resource.setrlimit(resource.RLIMIT_AS, previous_memory_limits)

        # Copy the results, then release optimization memory before saving.
        dicttest['x_opt_dim_30'] = x_best.tolist()
        dicttest['f_opt_dim_30'] = float(f_best)
        namespace.clear()
        del optimizer
        del landscape
        del x_best
        gc.collect()

        # Checkpoint the current record immediately.
        save_jsonl(file_path, records)

        print(
            f"Line {line_number} saved with best function value: "
            f"{f_best} and evaluations: {evaluations}"
        )
