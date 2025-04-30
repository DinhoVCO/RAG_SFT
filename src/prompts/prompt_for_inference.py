from prompts.boolq_prompts import format_input_context_boolq
from prompts.clapnq_prompts import format_input_context_clapnq
from prompts.covid_prompts import format_input_context_covid
from prompts.teleqna_prompts import format_input_context_teleqna

def prompt_for_inference(dataset_name, row, context=None):
    if(dataset_name=='covid'):
        return format_input_context_covid(row, context)
    elif(dataset_name=='teleqna'):
        return format_input_context_teleqna(row, context)
    elif(dataset_name=='clapnq'):
        return format_input_context_clapnq(row, context)
    elif(dataset_name=='boolq'):
        return format_input_context_boolq(row, context)
    else:
        raise ValueError(f"Incorrect dataset name: {dataset_name}")