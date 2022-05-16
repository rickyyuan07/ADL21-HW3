from datasets import Dataset


'''
    public.jsonl file format (len = 5494):
    'data_publish': '...',
    'title': the desired output of model (the label),
    'source_domain': ...,
    'maintext': the main text of the article (the input of model),
    'split': 'dev',
    'id': the unique id of the data point
'''
class T5_dataset(Dataset):
    def __init__(self, tokenizer, dataset, mode, input_length=1024, output_length=128, print_text=False):
        self.mode = mode
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.input_length = input_length
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = text.replace('\n','')
        text = text.replace('`', '')
        text = text.replace('"', '')
        text = text.replace("\r", "")
        
        return text
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['maintext']))
        
        input_ = self.clean_text(example_batch['maintext'])
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        if self.mode == 'test':
            return source
        
        target_ = self.clean_text(example_batch['title'])
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        return source, targets
  
    def __getitem__(self, index):
        if self.mode == 'test':
            source = self.convert_to_features(self.dataset[index])
            source_ids = source.input_ids.squeeze()
            src_mask    = source["attention_mask"].squeeze()
        else:
            source, targets = self.convert_to_features(self.dataset[index])
            source_ids = source.input_ids.squeeze()
            target_ids = targets.input_ids.squeeze()
            src_mask    = source["attention_mask"].squeeze()
            target_mask = targets["attention_mask"].squeeze()
        
        if self.mode == 'train':
            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "target": self.dataset[index]['title'], "id": self.dataset[index]['id']}
        elif self.mode == 'dev':
            return {"source_ids": source_ids, "source_mask": src_mask, "target": self.dataset[index]['title'], "id": self.dataset[index]['id']}
        return {"source_ids": source_ids, "source_mask": src_mask, "id": self.dataset[index]['id']}
        