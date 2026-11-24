# review-sentiment-insights

Telling positive from negative Amazon product reviews, and finding what unhappy customers complain about, using the public amazon_polarity dataset.

The point is to do the classic text-classification project carefully: check for duplicate reviews leaking between train and test, beat a word-list baseline before claiming anything, read the model's real mistakes, and only report complaint topics a person can actually name.

**Status:** in progress (day 8 of 10). Results will go here once there are any.

## Planned approach

1. Download a pinned copy of the dataset, check its hash, sample it to laptop size, and remove reviews duplicated across train and test
2. Baselines: majority class and the VADER sentiment lexicon
3. Bag-of-words vs TF-IDF, then logistic regression, Naive Bayes and a linear SVM, with n-grams
4. The words that push each class, and 50 labelled mistakes
5. Complaint topics with NMF on the negative reviews
6. Sentence embeddings vs TF-IDF
7. Score the untouched test set once, at the end

## Stack

Python 3.12, uv, pandas, scikit-learn, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

[amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (Zhang, Zhao and LeCun, 2015, from McAuley and Leskovec's Amazon reviews), Apache-2.0. It isn't committed: a script downloads it and checks its hash.

## Quick start

```bash
uv sync
make data        # downloads 1.15 GB once, checks hashes, writes data/processed
make lint test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
