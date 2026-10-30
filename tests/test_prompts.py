import pytest
from src.prompts import get_class_prompt, build_chat_messages, format_sft_dataset_row

def test_get_class_prompt():
    prompt = get_class_prompt('ACGT', ['DNA', 'LTR', 'LINE'])
    assert 'Danio_rerio (zebrafish)' in prompt
    assert '<dna>ACGT</dna>' in prompt
    assert 'DNA, LINE, LTR' in prompt
    assert 'Answer with only the class name from the list, nothing else.' in prompt

def test_build_chat_messages():
    inference_msgs = build_chat_messages('Prompt here')
    assert len(inference_msgs) == 1
    assert inference_msgs[0] == {'role': 'user', 'content': 'Prompt here'}

    training_msgs = build_chat_messages('Prompt here', assistant_response='DNA')
    assert len(training_msgs) == 2
    assert training_msgs[0] == {'role': 'user', 'content': 'Prompt here'}
    assert training_msgs[1] == {'role': 'assistant', 'content': 'DNA'}

def test_format_sft_dataset_row():
    row = {'Sequence': 'GGCCAA', 'repeat_class': 'Satellite'}
    formatted = format_sft_dataset_row(row, allowed_classes=['Satellite', 'DNA'])
    assert 'messages' in formatted
    assert len(formatted['messages']) == 2
    assert '<dna>GGCCAA</dna>' in formatted['messages'][0]['content']
    assert formatted['messages'][1]['content'] == 'Satellite'
