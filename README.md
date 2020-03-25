# SPO2 Evaluation

Welcome aboard the SpO2 evaluation project to help relief of the Covid2019 pandemic.

## Goals

1) Build a dataset enabling us to evaluate and develop algorithms for spo2 estimation within 2% accuracy from a smartphone camera images.
2) Compare latest state-of-the-art method for spo2 estimation from smartphone camera images.
3) Develop and ship a method to estimate spo2 within 2% accuracy from a smartphone camera images.

## Outcomes

1) A dataset from evaluating and developing spo2 estimation algorithms.
	* The dataset should be available publicly to enable anyone to implement and test their algorithm.
2) A decision wether to ship a given algorithm (and why) or implement our own.

## What is the plan?

1) Decide how to analyze the results of the algorithm and which data to compile.

Again the method is describe in the [overleaf document](https://www.overleaf.com/read/kwfmchzmmgtm).
Reach out again with suggestions and improvements. The current method is implemented in [the spo2 repo](https://github.com/CoVital-Project/Spo2_evaluation)

2) Compile a dataset from algorithm evaluation with the necessary data. 

The technical specification of the dataset are described [here](https://www.overleaf.com/read/kwfmchzmmgtm). 
If you would like to help with the specification, reach out to me and I can give you access to the overleaf document.

3) URGENT: consent forms

I think we're at that point where we need someone to make sure we can get that data and that we won't be in legal problem over it. If someone has an office in their University or company that could check that out for us, it would be great, please be in touch! I'm thinking GDPR.

4) Implement the comparison and some baseline methods

@Yoni and I are reimplementing some algorithms for testing. Anyone is welcome to pitch in and add their own implementation (be it personal or from a paper). I'll add a list of implemented method and the reference paper in the root README of the project. One thing we need to decide is the output format of the algorithm to run the tests.

## Where are we at

We've done step 1 (but it's all flexible, if you have input on them go for it, nothing is set in stone) and are moving forward on step 4.

Step 2 is starting and we need someone to get onto step 3.

# Quick description of the packages:

Each package has a README describing the use.
