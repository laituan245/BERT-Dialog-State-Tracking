import json

SPLITS = ['train', 'dev', 'test']
for split in SPLITS:
    # Data
    fp = f'data/woz/{split}.json'
    data = json.load(open(fp, 'r'))
    # Original Data
    id2oridialog = {}
    original_fp = f'data/woz_original/woz_{split}_en.json'
    original_data = json.load(open(original_fp, 'r'))
    for dialog in original_data:
        id2oridialog[dialog['dialogue_idx']] = dialog
    # Restore original transcripts
    for d in data['dialogues']:
        original_d = id2oridialog[d['dialogue_id']]
        current_turns = d['turns']
        original_turns = original_d['dialogue']
        assert(len(current_turns) == len(original_turns))
        for ix, (c_turn, o_turn) in enumerate(zip(current_turns, original_turns)):
            assert(c_turn['turn_id'] == o_turn['turn_idx'])
            d['turns'][ix]['transcript'] = o_turn['transcript'].split(' ')

    # Output
    with open(fp, 'w+') as f:
        json.dump(data, f)
