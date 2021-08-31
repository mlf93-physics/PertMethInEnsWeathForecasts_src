from collections import OrderedDict
import numpy as np
from pathlib import Path
from src.params.params import *

def match_start_positions_to_ref_file(args=None, header_dict=None,
        positions=None):
    """Match the positions to their respective record of the reference
    dataseries."""

    ref_file_match = OrderedDict()
    
    # The file ids which shall be imported
    matched_file_ids = positions//int(header_dict['record_max_time']*
        header_dict['sample_rate']/dt)
    
    # The positions in the files that shall be imported
    positions_relative_to_record = positions %\
        int(header_dict['record_max_time']*header_dict['sample_rate']/dt)
    
    # Save the positions per record id to a dictionary
    for file_id in set(matched_file_ids):
        record_positions = np.where(matched_file_ids == file_id)[0]
        ref_file_match[file_id] = positions_relative_to_record[record_positions]
    
    return ref_file_match

def get_sorted_ref_record_names(args=None):
    # Get file paths
    ref_record_names = list(Path(args['path'], 'ref_data').glob('*.csv'))
    ref_files_sort_index = np.argsort([str(ref_record_name) for
        ref_record_name in ref_record_names])
    ref_record_names_sorted = [ref_record_names[i] for i in ref_files_sort_index]

    return ref_record_names_sorted