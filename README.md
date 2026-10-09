# 🎰 EuroMillions Predictions

> A Python data analysis project for exploring EuroMillions historical results, probability, combinatorics and number-selection strategies.

---

## 📌 About the Project

Have you ever wondered whether some lottery combinations are more likely to appear than others?

From a mathematical perspective, a combination such as:

```text
01 - 02 - 03 - 04 - 05
```

has exactly the same probability of being drawn as any other valid combination.

However, lottery players often use different strategies when choosing their numbers. Some rely on personal patterns, lucky numbers, dates, dreams or other beliefs.

This project takes a different approach:

**What happens if we analyze the historical data instead?**

This project uses **Python, statistics, probability and combinatorics** to analyze EuroMillions draw history and generate number selections based on patterns found in the available data.

> ⚠️ **Important:** This project does **not** predict the future and cannot increase the mathematical probability of winning a lottery draw. Each draw is independent and contains an intrinsic random element.

The main purpose of the project is educational: **using a real-world dataset to practice data analysis, probability and Python programming.**

---

## 🎯 Objectives

The project aims to:

* Analyze historical EuroMillions results.
* Calculate the frequency of numbers appearing in previous draws.
* Identify statistical patterns in historical data.
* Apply probability and combinatorics concepts.
* Generate ranked number selections.
* Build combinations according to different mathematical criteria.
* Test the strategy against historical draws.
* Measure how often the generated selections contained winning numbers.
* Continuously improve the analysis as new results become available.

---

## 🧠 How Does It Work?

The system combines several concepts from probability and statistics.

### 1. Historical Data

The program analyzes previous EuroMillions draws and extracts information such as:

* Main numbers.
* Lucky Stars.
* Number frequency.
* Recent appearances.
* Historical distribution.
* Odd/even distribution.
* Low/high number distribution.

The historical information is then used to calculate different statistics for each number.

---

### 2. Law of Large Numbers

One of the concepts considered by the project is the **Law of Large Numbers**.

For example, imagine tossing a fair coin 10 times.

You could theoretically obtain:

```text
Heads: 8
Tails: 2
```

That doesn't mean the coin is biased.

If you repeat the experiment thousands of times, the observed proportion will generally become closer to the theoretical probability:

```text
Heads ≈ 50%
Tails ≈ 50%
```

Lottery data can be explored in a similar way.

However, this **does not mean that historical frequency predicts the next draw**. Previous draws do not influence future independent draws.

The project uses these statistical concepts as an analytical framework rather than as a guarantee of prediction.

---

## 🔢 Combinatorics

The system also considers different ways of constructing combinations.

For example, when selecting five numbers, the program can classify combinations according to their odd/even distribution:

| Odd | Even |
| --: | ---: |
|   5 |    0 |
|   4 |    1 |
|   3 |    2 |
|   2 |    3 |
|   1 |    4 |
|   0 |    5 |

The same type of analysis can be applied to other characteristics, such as:

* Low vs. high numbers.
* Number ranges.
* Frequency groups.
* Historical distributions.

These criteria are then used to construct and rank candidate combinations.

---

## 🎲 Number Selection

The program can work in two different ways.

### Manual Selection

You provide a number of values that you want to include.

The system then generates combinations using those numbers.

### Automatic Selection

The system analyzes the historical dataset and selects a group of candidate numbers automatically.

In many cases, the system selects approximately **25 numbers out of the 50 available main numbers**, ranking them according to the calculated criteria.

The resulting numbers are displayed in order of recommendation, allowing the user to either:

* Choose numbers manually.
* Let the program generate the combinations automatically.

---

## 📊 Combination Generation

The system generates a minimum number of combinations based on the selected number pool and the combinatorial rules.

For example, if a pool of 10 numbers is selected, the program can generate the possible 5-number combinations and filter them according to the configured criteria.

The purpose is not to claim that one combination is mathematically more likely to occur than another.

Instead, the system attempts to construct a **structured selection strategy** based on the statistical characteristics being analyzed.

---

## 🧪 Backtesting

One of the most interesting parts of the project is the historical testing.

The system can simulate the strategy against previous draws.

For example:

```text
Historical draws analyzed: 100

Draws containing ≥ 1 recommended number: XX
Draws containing ≥ 2 recommended numbers: XX
Draws containing ≥ 3 recommended numbers: XX
Draws containing ≥ 4 recommended numbers: XX
Draws containing ≥ 5 recommended numbers: XX
```

The idea is to evaluate how the strategy would have behaved using historical data.

To reduce overfitting, the system can exclude a portion of the most recent draws from the dataset used to calculate the recommendations and then test the recommendations against those unseen draws.

---

## 🛠️ Technologies

This project is primarily developed with:

* 🐍 **Python**
* 📊 **Pandas**
* 🔢 **NumPy**
* 📐 Probability & Combinatorics
* 📈 Statistical Analysis
* 🧪 Historical Backtesting

The project is also part of my personal development as I learn more about:

* Data Analysis
* Data Processing
* Statistics
* Probability
* Python
* ETL concepts
* Working with real-world datasets

---

## ▶️ Usage

Run the main program:

```bash
python main.py
```

The program will analyze the available historical data and generate the corresponding number recommendations and combinations.

Depending on the configuration, the user can either provide part of the number selection manually or allow the system to generate the complete selection automatically.

---

## 📈 Example Output

A typical execution can produce information similar to:

```text
Recommended numbers:

07
14
19
23
31
37
42
45
...

Recommended combinations:

07 - 14 - 23 - 31 - 42
07 - 19 - 23 - 37 - 45
14 - 19 - 31 - 37 - 42
...

Historical analysis:

Draws analyzed: 100
Draws with at least 1 recommended number: XX
Draws with at least 2 recommended numbers: XX
Draws with at least 3 recommended numbers: XX
```

---

## ⚠️ Important Disclaimer

This project is **not a lottery prediction machine**.

There is no statistical model capable of guaranteeing the winning numbers of a genuinely random lottery.

Historical results can be analyzed, patterns can be identified and combinations can be generated according to different criteria, but none of these methods changes the fundamental probability of an individual combination being drawn.

The results of previous draws **do not guarantee future results**.

This project should therefore be considered an **educational data-analysis experiment**, not a system for generating guaranteed lottery winnings.

---

## 💰 Responsible Gambling

If you choose to play the lottery:

* Only spend money you can afford to lose.
* Do not treat lottery games as an investment.
* Do not chase losses.
* Remember that statistical analysis cannot eliminate randomness.
* Set a personal budget and stick to it.

The purpose of this project is primarily to **learn Python, statistics and data analysis while working with an interesting real-world dataset.**

---

## 📚 What I Learned

This project has allowed me to practice several concepts that are useful beyond lottery analysis:

* Data cleaning and preparation.
* Exploratory data analysis.
* Statistical calculations.
* Probability.
* Combinatorics.
* Python programming.
* Working with Pandas and NumPy.
* Automating repetitive calculations.
* Historical backtesting.
* Evaluating the limitations of statistical models.
* Working with real-world datasets.

The lottery is simply the dataset.

**The real objective is learning how to turn raw data into information.** 📊🐍

---

## 🔮 Future Improvements

Possible future improvements include:

* [ ] Improve the data ingestion pipeline.
* [ ] Add automated retrieval of new EuroMillions results.
* [ ] Store historical data in a database.
* [ ] Add more statistical indicators.
* [ ] Improve the backtesting system.
* [ ] Add data visualizations.
* [ ] Build an interactive dashboard.
* [ ] Compare different selection strategies.
* [ ] Add machine-learning experiments for educational purposes.
* [ ] Improve test coverage.
* [ ] Automate the complete analysis pipeline.

---

## 📜 License

This project is intended for educational and personal use.

Use it responsibly and remember:

> **There is no algorithm that can remove randomness from a lottery.**

🎰 **Analyze the data. Learn from it. Play responsibly.**
