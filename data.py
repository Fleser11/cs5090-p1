from datasets import load_dataset


dataset_name = "codeparrot/codeparrot-clean"
dataset = load_dataset(dataset_name, split='train', streaming=True)
dataset.shuffle(seed=100, buffer_size=10000)

datasetIterator = iter(dataset)

allowed_licenses = ["apache-2.0", "mit", "bsd-3-clause"]
i = 10000
with open("data/tokenizer_text.txt", "w") as f:
    while i > 0:
        repo = next(datasetIterator)
        if repo["license"] in allowed_licenses:
            f.write(f"{repo['content']}\n")
            i = i - 1


