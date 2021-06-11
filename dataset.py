import json
import numpy as np

class Turn:
    def __init__(self, turn_id, transcript, turn_label, belief_state, system_acts, system_transcript, asr=None, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.asr = asr or []
        self.num = num or {}

    def to_dict(self):
        return {'turn_id': self.id,
                'transcript': self.transcript,
                'turn_label': self.turn_label,
                'belief_state': self.belief_state,
                'system_acts': self.system_acts,
                'system_transcript': self.system_transcript,
                'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class Dialogue:
    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id,
                'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

class Dataset:
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    def evaluate_preds(self, preds):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request'])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                pred_inform_list = list(pred_inform); pred_inform_list.sort()
                for s, v in pred_inform_list:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        if b['act'] != 'request':
                            gold_recovered.add((b['act'], fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}

class Ontology:
    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
