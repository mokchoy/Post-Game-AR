import csv

# Dictionary mapped exactly from image_99ed42.png
translation_map = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過度切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service"
}

def translate_csv_column(input_filename, output_filename, target_column):
    with open(input_filename, mode='r', encoding='utf-8') as infile, \
         open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        for row in reader:
            original_value = row.get(target_column, "").strip()
            # Replace if a match is found in the dictionary
            if original_value in translation_map:
                row[target_column] = translation_map[original_value]
            writer.writerow(row)

# Example usage: Update filenames and 'Target_Column_Name' to match your CSV
translate_csv_column(r'set-1\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\set1.csv', r'translation_test\merp.csv', r'type')