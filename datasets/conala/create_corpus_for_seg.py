import json

if __name__ == '__main__':
    with open('data/conala/all_text.txt', 'w', encoding='utf-8') as text_out, \
        open('data/conala/all_code.txt', 'w', encoding='utf-8') as code_out:
        with open('data/conala/conala-mined.jsonl', encoding='utf-8') as mined_file:
            for line in mined_file:
                obj = json.loads(line)
                text_out.write(obj['intent'].strip() + '\n')
                code_out.write(obj['snippet'].strip() + '\n')
        with open('data/conala/conala-train.json', encoding='utf-8') as train_file:
            for obj in json.load(train_file):
                if obj['rewritten_intent']:
                    text_out.write(obj['rewritten_intent'].strip() + '\n')
                else:
                    text_out.write(obj['intent'].strip() + '\n')
                code_out.write(obj['snippet'].strip() + '\n')

