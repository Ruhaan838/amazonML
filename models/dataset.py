# 23 grams -> tokenize
# predicted output: unique_id(index), "x_unit" ("12 grams")
# actual output: compare with actual csv file with column unique id and entity_value

# custom dataset: only needs unique id and value (not entity_name, etc. columns)

# image_link -> list = df['image_link'] :
# entity_name, entity_value

# new custom dataset -> image (c * h * w px) , group id, entity_value(gm, ml)

from torch.data.utils import Dataset, DataLoader

import os
import pandas as pd
import multiprocessing
import time

from tqdm import tqdm

from functools import partial
import urllib
from PIL import Image

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        # print(f"Error creating placeholder image: {e}")
        pass

def download_image(row, save_folder, retries=3, delay=3):
    image_link, group_id, entity_value = row['image_link'], row['group_id'], row['entity_value']

    if not isinstance(image_link, str):
        return

    filename = f"{group_id}_{entity_value}.jpg"
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            time.sleep(delay)

    create_placeholder_image(image_save_path)


def download_images_from_csv(csv_file, download_folder, allow_multiprocessing=True):
    df = pd.read_csv(csv_file)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)
        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(download_image_partial, df.to_dict('records')), total=len(df)))
            pool.close()
            pool.join()
    else:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            download_image(row, save_folder=download_folder,     retries=3, delay=3)


csv_file = r'..\amazonML\dataset\train.csv'
download_folder = 'Images'
download_images_from_csv(csv_file, download_folder)




class AmazonDataset(Dataset):
    def __init__(self, new_image_csv: any, image_dir: str, tokenizer , transform: any, seq_len:int = 20) -> None:
        self.new_image_csv = new_image_csv
        self.image_dir = image_dir

        self.image_list = os.listdir(self.image_dir)
        self.value_list = [new_image_csv[new_image_csv['image_link'] == self.image_list[i]]['entity_value'] for i in
                           range(len(self.image_dir))]
        self.tokenizer = tokenizer
        self.transform = transform
        self.sos = torch.tensor([tokenizer.token_to_id('<|start_of_sent|>')], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer.token_to_id('<|end_of_sent|>')], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer.token_to_id('<|padding|>')], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx):
        image = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image).convert('RGB')

        value = self.value_list[idx]

        value_token = self.tokenizer.encode(value).ids

        num_pad = self.pad - len(value_token) - 1

        if self.transform:
            image = self.transform(image)

        lstm_input = torch.cat([
            self.sos,
            torch.tensor(value_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * num_pad, dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(value_token, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad_token] * num_pad, dtype=torch.int64),
        ])

        return {
            'image': image,
            'label': label,
            'lstm_input': lstm_input,
        }

# conv1(64) -> ..... -> conv n(512)(a,b,c)  --> flatten (a, b*c) --> bi-dirxn lstm() --> bi-dirxn lstm() --> softmax

# image_path , entity_value
# test.csv -> group_id -> image_link ->

# train.csv
# image_link,group_id,entity_name,entity_value
# https://m.media-amazon.com/images/I/61I9XdN6OFL.jpg,748919,item_weight,500.0 gram
# https://m.media-amazon.com/images/I/71gSRbyXmoL.jpg,916768,item_volume,1.0 cup
# https://m.media-amazon.com/images/I/61BZ4zrjZXL.jpg,459516,item_weight,0.709 gram
# https://m.media-amazon.com/images/I/612mrlqiI4L.jpg,459516,item_weight,0.709 gram

# test.csv
# index,image_link,group_id,entity_name,pred:entity_value
# 0,https://m.media-amazon.com/images/I/110EibNyclL.jpg,156839,height
# 1,https://m.media-amazon.com/images/I/11TU2clswzL.jpg,792578,width
# 2,https://m.media-amazon.com/images/I/11TU2clswzL.jpg,792578,height
# 3,https://m.media-amazon.com/images/I/11TU2clswzL.jpg,792578,depth
# 4,https://m.media-amazon.com/images/I/11gHj8dhhrL.jpg,792578,depth
# 5,https://m.media-amazon.com/images/I/11gHj8dhhrL.jpg,792578,height

# sample_test_out.csv
# index,prediction
# 0,21.9 foot
# 1,10 foot
# 2,
# 3,289.52 kilovolt
# 4,1078.99 kilowatt
# 5,58.21 ton

